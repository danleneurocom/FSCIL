#!/usr/bin/env python3
"""
oneclass_capsule_gpu_amp_contrastive_safe.py

One‐class anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
Same as before, but avoids OOM by encoding support images one by one.
"""

import os, random, argparse, numpy as np
from multiprocessing import freeze_support

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import faiss

ImageFile.LOAD_TRUNCATED_IMAGES = True
from segment_anything import sam_model_registry

# ─── Dataset definitions ────────────────────────────────────────────────
class GoodTrainDataset(Dataset):
    def __init__(self, data_root, transform=None):
        folder = os.path.join(data_root, "train", "good")
        if not os.path.isdir(folder):
            raise RuntimeError(f"Expected folder: {folder}")
        self.samples = [os.path.join(folder, fn)
                        for fn in sorted(os.listdir(folder))
                        if fn.lower().endswith(".png") and not fn.startswith(".")]
        if not self.samples:
            raise RuntimeError("No samples in train/good/")
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, 0

class TestDataset(Dataset):
    def __init__(self, data_root, transform=None):
        base = os.path.join(data_root, "test")
        if not os.path.isdir(base):
            raise RuntimeError(f"Expected folder: {base}")
        self.samples = []
        for cls in sorted(os.listdir(base)):
            if cls.startswith("."): continue
            fld = os.path.join(base, cls)
            if not os.path.isdir(fld): continue
            lbl = 0 if cls=="good" else 1
            for fn in sorted(os.listdir(fld)):
                if fn.lower().endswith(".png") and not fn.startswith("."):
                    self.samples.append((os.path.join(fld, fn), lbl))
        if not self.samples:
            raise RuntimeError("No samples in test/")
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, lbl

# ─── Sampling helpers ───────────────────────────────────────────────────
def sample_mixed_support(good_idxs, defect_idxs, n_shot):
    goods = random.sample(good_idxs,  n_shot)
    bads  = random.sample(defect_idxs, n_shot)
    return goods, bads

def sample_good_query(good_idxs, n_query):
    return random.sample(good_idxs, n_query)

def sample_defect_query(defect_idxs, n_query):
    return random.sample(defect_idxs, n_query)

def sample_test_episode(ds, test_map, n_query):
    goods = random.sample(test_map[0], n_query)
    bads  = random.sample(test_map[1], n_query)
    idxs  = goods + bads
    imgs, lbls = zip(*(ds[i] for i in idxs))
    return torch.stack(imgs), torch.tensor(lbls)

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--n_shot",         type=int, default=5)
    p.add_argument("--n_query",        type=int, default=5)
    p.add_argument("--train_episodes", type=int, default=100)
    p.add_argument("--inner_steps",    type=int, default=5)
    p.add_argument("--eval_episodes",  type=int, default=100)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    # transforms
    tf = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std =[0.26862954,0.26130258,0.27577711],
        ),
    ])

    # datasets
    train_ds    = GoodTrainDataset(args.data_root, transform=tf)
    test_ds     = TestDataset   (args.data_root, transform=tf)
    good_idxs   = list(range(len(train_ds)))
    test_good   = [i for i,(_,lbl) in enumerate(test_ds.samples) if lbl==0]
    test_defect = [i for i,(_,lbl) in enumerate(test_ds.samples) if lbl==1]
    test_map    = {0: test_good, 1: test_defect}

    print(f"# train good   = {len(good_idxs)}")
    print(f"# test good    = {len(test_good)}, defect = {len(test_defect)}")

    # load & freeze SAM
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam = sam.to(device).eval()
    for p_ in sam.parameters():
        p_.requires_grad = False

    # encode single-batch
    def encode(x):
        fmap = sam.image_encoder(x)       # [B,C,64,64]
        feat = fmap.mean(dim=(2,3))       # [B,C]
        return F.normalize(feat, dim=1)   # [B,C]

    # encode sequence to save memory
    def encode_seq(imgs):
        feats = []
        for img in imgs:
            feats.append(encode(img.unsqueeze(0)))  # [1,C]
        return torch.cat(feats, 0)  # [N,C]

    # prompt params
    C     = sam.image_encoder.neck[0].out_channels
    slow  = nn.Parameter(torch.randn(4,  C, device=device)*0.02)
    proto = nn.Parameter(torch.randn(1,  C, device=device)*0.02)
    fast  = nn.Parameter(torch.randn(2,  C, device=device)*0.02)

    opt_f  = optim.SGD([fast],        lr=1e-2)
    opt_sp = optim.SGD([slow, proto], lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    margin_in  = 0.1
    margin_out = 0.2

    def make_prompt(fast_tensor):
        p = proto + slow.mean(0, keepdim=True) + fast_tensor.mean(0, keepdim=True)
        return F.normalize(p, dim=1)  # [1,C]

    # ─── Meta-training ───────────────────────────────────────────────────
    print("\n>> Meta‐training with outlier exposure …")
    for ep in range(1, args.train_episodes+1):
        goods, bads = sample_mixed_support(good_idxs, test_defect, args.n_shot)
        imgs_sup = torch.stack(
            [train_ds[i][0] for i in goods] +
            [test_ds[i][0]  for i in bads ]
        ).to(device)  # [2*n_shot,3,1024,1024]

        # inner‐loop: fast
        fast.data.zero_()
        for _ in range(args.inner_steps):
            opt_f.zero_grad()
            with autocast():
                fs    = encode_seq(imgs_sup)         # [2n_shot,C]
                pr    = make_prompt(fast)            # [1,C]
                sims  = (fs @ pr.T).squeeze(1)       # [2n_shot]
                pos, neg = sims[:args.n_shot], sims[args.n_shot:]
                loss_in  = F.relu(neg - pos + margin_in).mean()
            scaler.scale(loss_in).backward()
            scaler.step(opt_f); scaler.update()

        # outer‐loop: slow+proto
        fast.requires_grad_(False)
        opt_sp.zero_grad()
        with autocast():
            qg = sample_good_query(good_idxs, args.n_query)
            fq = encode_seq(torch.stack([train_ds[i][0] for i in qg]).to(device))
            qd = sample_defect_query(test_defect, args.n_query)
            fd = encode_seq(torch.stack([test_ds[i][0] for i in qd]).to(device))
            pr = make_prompt(fast)               # [1,C]
            sp, sn = (fq @ pr.T).squeeze(1), (fd @ pr.T).squeeze(1)
            loss_out = F.relu(sn - sp + margin_out).mean()
        scaler.scale(loss_out).backward()
        scaler.step(opt_sp); scaler.update()
        fast.requires_grad_(True)

        if ep == 1 or ep % 10 == 0:
            print(f"[EP {ep:03d}/{args.train_episodes:03d}] "
                  f"inner {loss_in:.4f}, outer {loss_out:.4f}")

    # ─── Build GPU FAISS index on normals ────────────────────────────────
    print("\n>> Building FAISS index for eval …")
    all_feats = []
    with torch.no_grad():
        for i in good_idxs:
            img,_ = train_ds[i]
            all_feats.append(encode(img.unsqueeze(0).to(device)).cpu().numpy())
    mem = np.vstack(all_feats).astype(np.float32)
    res    = faiss.StandardGpuResources()
    cpu_idx= faiss.IndexFlatL2(mem.shape[1])
    gpu_idx= faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    gpu_idx.add(mem)

    # ─── Few-shot evaluation ─────────────────────────────────────────────
    print("\n>> Few‐shot evaluation …")
    accs = []
    for _ in tqdm(range(args.eval_episodes), desc="Eval"):
        # adapt on mixed support
        goods, bads = sample_mixed_support(good_idxs, test_defect, args.n_shot)
        imgs_sup = torch.stack(
            [train_ds[i][0] for i in goods] +
            [test_ds[i][0]  for i in bads ]
        ).to(device)

        fast_eval = fast.clone().detach().requires_grad_(True)
        opt_feval  = optim.SGD([fast_eval], lr=1e-2)
        for _ in range(args.inner_steps):
            opt_feval.zero_grad()
            with autocast():
                fs    = encode_seq(imgs_sup)
                pr    = make_prompt(fast_eval)
                sims  = (fs @ pr.T).squeeze(1)
                pos, neg = sims[:args.n_shot], sims[args.n_shot:]
                loss_in  = F.relu(neg - pos + margin_in).mean()
            scaler.scale(loss_in).backward()
            scaler.step(opt_feval); scaler.update()

        # threshold on positives
        with torch.no_grad():
            fs   = encode_seq(imgs_sup)
            pr   = make_prompt(fast_eval).squeeze(0)
            dpos = (fs[:args.n_shot] - pr).pow(2).sum(1).sqrt()
            thr  = dpos.max().item()

        # query & classify
        imgs_q, lbls_q = sample_test_episode(test_ds, test_map, args.n_query)
        flat = encode_seq(imgs_q.to(device)).cpu().numpy().astype(np.float32)
        d2   = gpu_idx.search(flat, 1)[0].reshape(-1)
        preds = (d2 > thr).astype(int)
        accs.append((preds == lbls_q.numpy()).mean())

    m, s = np.mean(accs), np.std(accs)
    print(f"\nFinal few‐shot accuracy: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
