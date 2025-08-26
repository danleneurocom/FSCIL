#!/usr/bin/env python3
"""
oneclass_capsule_ucad.py

Enhanced one-class anomaly detection on MVTec “capsule” with:
 - Multi-layer patch features from SAM
 - Multi-scale ensemble
 - GPU-accelerated coreset sampling with progress
 - GPU FAISS index for k-NN scoring (chunked to avoid GPU OOM)
 - Fast/slow meta-tuning of prompts (normalized)
 - Logging of inner-loop support losses
 - Threshold calibration (percentile)
 - AMP on GPU
"""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import freeze_support

import torch
import faiss
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

# allow truncated PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

from segment_anything import sam_model_registry

# ─── Dataset ────────────────────────────────────────────────────────────
class GoodTrainDataset(Dataset):
    """Good-only training images (train/good/*.png)."""
    def __init__(self, data_root, transform):
        folder = os.path.join(data_root, "train", "good")
        if not os.path.isdir(folder):
            raise RuntimeError(f"Expected folder: {folder}")
        self.transform = transform
        self.samples = sorted([
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith(".png") and not fn.startswith(".")
        ])
        if not self.samples:
            raise RuntimeError(f"No training samples found under {folder}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        return self.transform(img), 0

class TestDataset(Dataset):
    """Test images (test/good & test/defect)."""
    def __init__(self, data_root, transform):
        base = os.path.join(data_root, "test")
        if not os.path.isdir(base):
            raise RuntimeError(f"Expected folder: {base}")
        self.transform = transform
        self.samples = []
        for cls in sorted(os.listdir(base)):
            if cls.startswith("."): 
                continue
            folder = os.path.join(base, cls)
            if not os.path.isdir(folder):
                continue
            lbl = 0 if cls == "good" else 1
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(".png") and not fn.startswith("."):
                    self.samples.append((os.path.join(folder, fn), lbl))
        if not self.samples:
            raise RuntimeError(f"No test samples found under {base}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), lbl

# ─── Simplified SAM patch-extractor ─────────────────────────────────────
class MultiLayerSAM(nn.Module):
    def __init__(self, sam: nn.Module):
        super().__init__()
        self.conv_proj = sam.image_encoder.patch_embed.proj

    def forward(self, x: torch.Tensor):
        fmap = self.conv_proj(x)  # [B, C, Hp, Wp]
        return [fmap]

# ─── GPU-tensor coreset sampler ─────────────────────────────────────────
class ApproximateGreedyCoresetSampler:
    def __init__(self, frac, device):
        self.frac   = frac
        self.device = torch.device(device)

    def subsample(self, features: np.ndarray):
        N, D = features.shape
        m    = max(1, int(self.frac * N))
        dev_feats = torch.from_numpy(features).to(self.device)  # [N, D]

        selected = [random.randrange(N)]
        dist2     = ((dev_feats - dev_feats[selected[0]].unsqueeze(0))**2).sum(dim=1)

        print(f">>> Starting GPU coreset sampling on {N} vectors, target {m}")
        for i in range(1, m):
            if i < 10 or i % 500 == 0:
                print(f"    coresamp: selected {i}/{m}…")
            nxt   = int(dist2.argmax().item())
            selected.append(nxt)
            new_d2 = ((dev_feats - dev_feats[nxt].unsqueeze(0))**2).sum(dim=1)
            dist2  = torch.min(dist2, new_d2)

        return features[selected]

# ─── RescaleSegmentor ───────────────────────────────────────────────────
class RescaleSegmentor:
    def __init__(self, target_size):
        import torch.nn.functional as F
        self.F = F; self.target = target_size
    def upsample(self, patch_scores: torch.Tensor):
        x = patch_scores.unsqueeze(1)  # [B,1,S,S]
        x = self.F.interpolate(x,
                               size=self.target,
                               mode="bilinear",
                               align_corners=False)
        return x[:,0]  # [B,H,W]

# ─── Episode Samplers ───────────────────────────────────────────────────
def sample_good_episode(idxs, n_shot, n_query):
    pick = random.sample(idxs, n_shot + n_query)
    return pick[:n_shot], pick[n_shot:]
def sample_test_episode(idx_map, n_query):
    return random.sample(idx_map[0], n_query) + random.sample(idx_map[1], n_query)

# ─── Helper: chunked GPU search ─────────────────────────────────────────
def batched_gpu_search(index, queries: np.ndarray, k: int,
                       batch_size: int = 512) -> np.ndarray:
    """
    queries: (Nq, D) numpy array.
    returns: (Nq, k) distances
    """
    Nq, _ = queries.shape
    all_d = np.empty((Nq, k), dtype=np.float32)
    for i0 in range(0, Nq, batch_size):
        chunk = queries[i0:i0+batch_size]
        d_chunk = index.search(chunk, k)[0]  # (chunk_size, k)
        all_d[i0:i0+batch_size] = d_chunk
    return all_d

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--n_shot",         type=int, default=5)
    parser.add_argument("--n_query",        type=int, default=5)
    parser.add_argument("--train_episodes", type=int, default=50)
    parser.add_argument("--inner_steps",    type=int, default=5)
    parser.add_argument("--eval_episodes",  type=int, default=100)
    parser.add_argument("--scales",         type=int, nargs="+", default=[1024,512])
    parser.add_argument("--memory_frac",    type=float, default=0.1)
    parser.add_argument("--k_nn",           type=int,   default=1)
    parser.add_argument("--threshold_percentile", type=float, default=100.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
          mean=[0.48145466,0.4578275,0.40821073],
          std =[0.26862954,0.26130258,0.27577711],
        ),
    ])

    # ─── Build memory bank ───────────────────────────────────────────────
    print(">> Building normal-patch memory bank …")
    sam_raw = sam_model_registry["vit_b"](
        checkpoint=args.sam_checkpoint).to(device).eval()
    mlsam   = MultiLayerSAM(sam_raw).to(device)
    sampler = ApproximateGreedyCoresetSampler(args.memory_frac, args.device)

    all_feats = []
    for scale in args.scales:
        tf     = transforms.Compose([transforms.Resize((scale,scale)), base_tf])
        ds     = GoodTrainDataset(args.data_root, tf)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
        for imgs, _ in tqdm(loader, desc=f"scale {scale}"):
            imgs = imgs.to(device)
            with torch.no_grad():
                for fmap in mlsam(imgs):
                    B,C,Hp,Wp = fmap.shape
                    patches  = fmap.flatten(2).permute(0,2,1).reshape(-1,C)
                    all_feats.append(patches.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    print(f"  » total patches: {all_feats.shape[0]}")

    memory_feats = sampler.subsample(all_feats)
    print(f"  » memory size: {memory_feats.shape}")

    # ─── GPU FAISS index for inference ───────────────────────────────────
    res     = faiss.StandardGpuResources()
    cpu_idx = faiss.IndexFlatL2(memory_feats.shape[1])
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    gpu_idx.add(memory_feats.astype(np.float32))

    # ─── Prompts & Meta-Tune ────────────────────────────────────────────
    D     = memory_feats.shape[1]
    fast  = nn.Parameter(torch.randn(2, D, device=device)*0.02)
    slow  = nn.Parameter(torch.randn(4, D, device=device)*0.02)
    proto = nn.Parameter(torch.randn(1, D, device=device)*0.02)

    opt_f  = optim.SGD([fast],        lr=1e-1)
    opt_s  = optim.SGD([slow, proto], lr=1e-2)
    mse    = nn.MSELoss()
    scaler = GradScaler()

    def pool_feats(img, scale):
        x    = transforms.Resize((scale,scale))(img)
        fmap = mlsam(x)[0]  # only one fmap
        B,C,Hp,Wp = fmap.shape
        patches   = fmap.flatten(2).permute(0,2,1)  # [B, P, C]
        return patches

    print(">> Meta-training prompts …")
    good_ds    = GoodTrainDataset(args.data_root, base_tf)
    train_idxs = list(range(len(good_ds)))

    for epi in range(1, args.train_episodes+1):
        sup_i, qry_i = sample_good_episode(train_idxs,
                                           args.n_shot, args.n_query)

        # support
        sup_feats = []
        for i in sup_i:
            img = good_ds[i][0].unsqueeze(0).to(device)
            for s in args.scales:
                with torch.no_grad():
                    sup_feats.append(pool_feats(img, s))
        sup_feats = torch.cat(sup_feats, dim=1)      # [1, P, D]
        sup_feats = F.normalize(sup_feats, dim=2)

        # query
        qry_feats = []
        for i in qry_i:
            img = good_ds[i][0].unsqueeze(0).to(device)
            for s in args.scales:
                with torch.no_grad():
                    qry_feats.append(pool_feats(img, s))
        qry_feats = torch.cat(qry_feats, dim=1)
        qry_feats = F.normalize(qry_feats, dim=2)

        # inner-loop adaptation + logging
        fast.data.zero_()
        print(f"[EP {epi:02d}] inner-loop sup-losses:", end="")
        for step in range(args.inner_steps):
            opt_f.zero_grad()
            with autocast("cuda"):
                prompt = proto + slow.mean(0,True) + fast.mean(0,True)
                prompt = prompt.unsqueeze(1)        # [1,1,D]
                prompt = F.normalize(prompt, dim=2)
                loss_sup = mse(sup_feats,
                               prompt.expand_as(sup_feats))
            scaler.scale(loss_sup).backward()
            scaler.step(opt_f)
            scaler.update()
            print(f" {loss_sup.item():.4f}", end="\n" if step+1==args.inner_steps else "")

        # outer update on query
        fast.requires_grad_(False)
        opt_s.zero_grad()
        with autocast("cuda"):
            prompt = proto + slow.mean(0,True) + fast.mean(0,True)
            prompt = F.normalize(prompt.unsqueeze(1), dim=2)
            loss_q = mse(qry_feats,
                         prompt.expand_as(qry_feats))
        scaler.scale(loss_q).backward()
        scaler.step(opt_s)
        scaler.update()
        fast.requires_grad_(True)
        print(f"[EP {epi:02d}] meta-loss {loss_q.item():.4f}")

    # ─── Few-shot eval ───────────────────────────────────────────────────
    print("\n>> Few-shot eval …")
    test_ds = TestDataset(args.data_root, base_tf)
    idx_map = {0:[], 1:[]}
    for i, (_, lbl) in enumerate(test_ds.samples):
        idx_map[lbl].append(i)

    segor = RescaleSegmentor((1024,1024))
    accs  = []

    for _ in tqdm(range(args.eval_episodes), desc="eval"):
        fast_eval = fast.detach().clone().requires_grad_(True)
        opt_fe    = optim.SGD([fast_eval], lr=1e-2)

        # adapt on support
        sup_i     = random.sample(idx_map[0], args.n_shot)
        sup_feats = []
        for i in sup_i:
            img = test_ds[i][0].unsqueeze(0).to(device)
            for s in args.scales:
                with torch.no_grad():
                    sup_feats.append(pool_feats(img, s))
        sup_feats = torch.cat(sup_feats, dim=1)
        sup_feats = F.normalize(sup_feats, dim=2)

        for _ in range(args.inner_steps):
            opt_fe.zero_grad()
            with autocast("cuda"):
                prompt = proto + slow.mean(0,True) + fast_eval.mean(0,True)
                prompt = F.normalize(prompt.unsqueeze(1), dim=2)
                loss   = mse(sup_feats,
                             prompt.expand_as(sup_feats))
            scaler.scale(loss).backward()
            scaler.step(opt_fe)
            scaler.update()

        with torch.no_grad():
            prompt = proto + slow.mean(0,True) + fast_eval.mean(0,True)
            prompt = F.normalize(prompt.unsqueeze(1), dim=2)
            d_sup   = torch.norm(sup_feats - prompt, dim=2)
            thr     = np.percentile(
                d_sup.cpu().numpy(),
                args.threshold_percentile)

        # inference with chunked GPU FAISS
        qidx       = sample_test_episode(idx_map, args.n_query)
        imgs, lbls = zip(*(test_ds[i] for i in qidx))
        imgs       = torch.stack(imgs).to(device)

        all_scores = []
        for s in args.scales:
            batch   = transforms.Resize((s,s))(imgs)
            fmap    = mlsam(batch)[0]
            B,C,Hp,Wp = fmap.shape
            patches   = fmap.flatten(2).permute(0,2,1)  # [B, P, D]
            flat_q    = patches.cpu().numpy().reshape(-1, D).astype(np.float32)
            dists     = batched_gpu_search(gpu_idx, flat_q,
                                           args.k_nn, batch_size=512)
            D2        = dists.mean(axis=1)
            patch_scores = torch.from_numpy(D2).view(B, -1).to(device)
            all_scores.append(patch_scores)

        masks      = torch.stack([segor.upsample(ps)
                                  for ps in all_scores], 0).max(0)
        img_scores = masks.view(masks.shape[0], -1).max(1)[0].cpu().numpy()
        preds      = (img_scores > thr).astype(int)
        accs.append((preds == np.array(lbls)).mean())

    print(f"\nFinal few-shot anomaly accuracy: "
          f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

if __name__=="__main__":
    main()
