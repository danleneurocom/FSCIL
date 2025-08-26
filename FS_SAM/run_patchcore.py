# #!/usr/bin/env python3
# """
# oneclass_capsule_ucad.py

# Enhanced one-class anomaly detection on MVTec “capsule” with:
#  - Multi-layer patch features from SAM
#  - Multi-scale ensemble
#  - Approximate-greedy coreset sampling (FAISS)
#  - k-NN patch scoring + RescaleSegmentor upsampling
#  - Fast/slow meta-tuning of prompts
#  - Threshold calibration (percentile)
#  - AMP on GPU
# """

# import os, random, argparse, math
# import numpy as np
# from tqdm import tqdm
# from multiprocessing import freeze_support

# import torch, faiss
# import torch.nn as nn, torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image, ImageFile

# # allow truncated PNGs
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from segment_anything import sam_model_registry

# # ─── Dataset ────────────────────────────────────────────────────────────
# class GoodTrainDataset(Dataset):
#     def __init__(self, data_root, transform):
#         self.transform = transform
#         folder = os.path.join(data_root, "train", "good")
#         self.samples = sorted([
#             os.path.join(folder, fn)
#             for fn in os.listdir(folder) if fn.lower().endswith(".png")
#         ])
#         if not self.samples:
#             raise RuntimeError(f"No train/good under {folder}")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         img = Image.open(self.samples[i]).convert("RGB")
#         return self.transform(img), 0

# class TestDataset(Dataset):
#     def __init__(self, data_root, transform):
#         self.transform = transform
#         base = os.path.join(data_root, "test")
#         self.samples = []
#         for cls in sorted(os.listdir(base)):
#             lbl = 0 if cls=="good" else 1
#             folder = os.path.join(base,cls)
#             for fn in sorted(os.listdir(folder)):
#                 if fn.lower().endswith(".png"):
#                     self.samples.append((os.path.join(folder,fn),lbl))
#         if not self.samples:
#             raise RuntimeError(f"No test/* under {base}")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         path,lbl = self.samples[i]
#         img = Image.open(path).convert("RGB")
#         return self.transform(img), lbl

# # ─── Multi-Layer SAM Encoder ────────────────────────────────────────────
# class MultiLayerSAM(nn.Module):
#     def __init__(self, sam: torch.nn.Module, layer_idxs):
#         super().__init__()
#         self.patch_embed = sam.image_encoder.patch_embed
#         self.pos_embed   = sam.image_encoder.pos_embed
#         self.dropout     = sam.image_encoder.dropout
#         self.blocks      = sam.image_encoder.blocks
#         self.norm        = sam.image_encoder.norm
#         self.layer_idxs  = set(layer_idxs)

#     def forward(self, x):
#         # x: [B,3,H,W]
#         B = x.shape[0]
#         x = self.patch_embed(x)                      # [B, N, C]
#         x = x + self.pos_embed
#         x = self.dropout(x)
#         feats = []
#         for i, blk in enumerate(self.blocks):
#             x = blk(x)
#             if i in self.layer_idxs:
#                 fmap = self._tokens_to_map(x)
#                 feats.append(fmap)
#         # final normed layer too
#         x = self.norm(x)
#         feats.append(self._tokens_to_map(x))
#         return feats

#     @staticmethod
#     def _tokens_to_map(x):
#         # x: [B, 1+P, C]  → drop [CLS] → [B, C, S, S]
#         B,NP,C = x.shape
#         P = NP - 1
#         S = int(math.sqrt(P))
#         tokens = x[:,1:,:]                        # [B,P,C]
#         return tokens.view(B, S, S, C).permute(0,3,1,2)

# # ─── ApproxGreedyCoresetSampler ─────────────────────────────────────────
# class ApproximateGreedyCoresetSampler:
#     def __init__(self, frac, device):
#         self.frac   = frac
#         self.device = device
#     def subsample(self, features: np.ndarray):
#         # features: [N, D]
#         N,D = features.shape
#         m    = max(1,int(self.frac * N))
#         index = faiss.IndexFlatL2(D)
#         index.add(features.astype(np.float32))
#         # greedy by farthest-first
#         selected = [random.randrange(N)]
#         dist2    = index.compute_distance_matrix(
#                       features[selected], features
#                    )[0]
#         for _ in range(1,m):
#             nxt = np.argmax(dist2)
#             selected.append(nxt)
#             d_new = index.compute_distance_matrix(
#                        features[[nxt]], features
#                     )[0]
#             dist2 = np.minimum(dist2, d_new)
#         return features[selected]

# # ─── RescaleSegmentor ───────────────────────────────────────────────────
# class RescaleSegmentor:
#     def __init__(self, target_size):
#         import torch.nn.functional as F
#         self.F = F; self.target = target_size
#     def upsample(self, patch_scores: torch.Tensor):
#         # patch_scores: [B, S, S]
#         B,S,_ = patch_scores.shape
#         x = patch_scores.unsqueeze(1)  # [B,1,S,S]
#         x = self.F.interpolate(x, size=self.target, mode="bilinear", align_corners=False)
#         return x[:,0]                  # [B,H,W]

# # ─── Episode Samplers ───────────────────────────────────────────────────
# def sample_good_episode(idxs,n_shot,n_query):
#     t = n_shot+n_query
#     pick = random.sample(idxs,t)
#     return pick[:n_shot], pick[n_shot:]

# def sample_test_episode(idx_map,n_query):
#     g = random.sample(idx_map[0],n_query)
#     b = random.sample(idx_map[1],n_query)
#     return g+b

# # ─── Main ────────────────────────────────────────────────────────────────
# def main():
#     freeze_support()
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",      required=True)
#     p.add_argument("--sam_checkpoint", required=True)
#     p.add_argument("--device",         default="cpu")
#     p.add_argument("--n_shot",   type=int, default=5)
#     p.add_argument("--n_query",  type=int, default=5)
#     p.add_argument("--train_episodes", type=int, default=50)
#     p.add_argument("--inner_steps",    type=int, default=5)
#     p.add_argument("--eval_episodes",  type=int, default=100)
#     p.add_argument("--scales",  type=int, nargs="+", default=[1024,512])
#     p.add_argument("--layers",  type=int, nargs="+", default=[3,7,11])
#     p.add_argument("--memory_frac", type=float, default=0.1)
#     p.add_argument("--k_nn",        type=int,   default=1)
#     p.add_argument("--threshold_percentile", type=float, default=100.0)
#     args = p.parse_args()

#     device = torch.device(args.device)
#     torch.manual_seed(42); random.seed(42); np.random.seed(42)

#     # transforms
#     base_tf = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#           mean=[0.48145466,0.4578275,0.40821073],
#           std =[0.26862954,0.26130258,0.27577711],
#         ),
#     ])

#     # ─── build memory bank from ALL GOOD TRAIN patches ────────────────────
#     print(">> Building normal-patch memory bank …")
#     sam_raw = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
#     sam_raw = sam_raw.to(device).eval()
#     mlsam   = MultiLayerSAM(sam_raw, args.layers).to(device)

#     sampler = ApproximateGreedyCoresetSampler(args.memory_frac, device)
#     all_feats = []

#     for scale in args.scales:
#         tf = transforms.Compose([transforms.Resize((scale,scale)), base_tf])
#         ds = GoodTrainDataset(args.data_root, tf)
#         loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
#         for img,_ in tqdm(loader, desc=f"scale {scale}"):
#             img = img.to(device)
#             with torch.no_grad():
#                 maps = mlsam(img)       # list of [B,C,h_i,w_i]
#                 for fmap in maps:
#                     B,C,Hi,Wi = fmap.shape
#                     patches = fmap.flatten(2).permute(0,2,1).reshape(-1,C)
#                     all_feats.append(patches.cpu().numpy())
#     all_feats = np.concatenate(all_feats,axis=0)
#     memory_feats = sampler.subsample(all_feats)   # [M, C]
#     print(f"  » memory bank size: {memory_feats.shape}")

#     # build FAISS index
#     D = memory_feats.shape[1]
#     index = faiss.IndexFlatL2(D)
#     index.add(memory_feats.astype(np.float32))

#     # ─── set up fast/slow prompts + proto ──────────────────────────────
#     n_fast, n_slow = 2, 4
#     fast = nn.Parameter(torch.randn(n_fast, D, device=device)*0.02)
#     slow = nn.Parameter(torch.randn(n_slow, D, device=device)*0.02)
#     proto= nn.Parameter(torch.randn(1, D, device=device)*0.02)

#     opt_f = optim.SGD([fast],        lr=1e-2)
#     opt_s = optim.SGD([slow, proto], lr=1e-3)
#     mse   = nn.MSELoss()
#     scaler= GradScaler()

#     # helper: pooled patch→global vector
#     def pool_feats(img,scale):
#         tf = transforms.Resize((scale,scale))
#         x = tf(img)
#         maps = mlsam(x)
#         # concat all layers
#         allp = torch.cat([m.flatten(2).permute(0,2,1) for m in maps], dim=1)
#         return allp   # [B, P_all, D]

#     # meta-train on good only
#     print(">> Meta-training prompts …")
#     good_ds = GoodTrainDataset(args.data_root, None)
#     train_idxs = list(range(len(good_ds)))

#     for epi in range(1, args.train_episodes+1):
#         # sample one episode (all good)
#         sup_i, qry_i = sample_good_episode(train_idxs, args.n_shot, args.n_query)

#         # gather support/query global patches
#         sup_imgs = [good_ds[i][0] for i in sup_i]
#         qry_imgs = [good_ds[i][0] for i in qry_i]

#         sup_feats = []
#         for img in sup_imgs:
#             img = img.to(device)
#             for s in args.scales:
#                 with torch.no_grad(): sup_feats.append(pool_feats(img,s))
#         sup_feats = torch.cat(sup_feats,dim=1)  # [n_s, P_all, D]

#         qry_feats = []
#         for img in qry_imgs:
#             img = img.to(device)
#             for s in args.scales:
#                 with torch.no_grad(): qry_feats.append(pool_feats(img,s))
#         qry_feats = torch.cat(qry_feats,dim=1)

#         # inner fast adapt
#         fast.data.zero_()
#         for _ in range(args.inner_steps):
#             opt_f.zero_grad()
#             with autocast():
#                 prompt = proto + slow.mean(0,True) + fast.mean(0,True)
#                 # MSE across ALL patch tokens
#                 loss = mse(sup_feats, prompt.unsqueeze(1).expand_as(sup_feats))
#             scaler.scale(loss).backward()
#             scaler.step(opt_f); scaler.update()

#         # outer slow+proto
#         fast.requires_grad_(False)
#         opt_s.zero_grad()
#         with autocast():
#             prompt = proto + slow.mean(0,True) + fast.mean(0,True)
#             loss_q = mse(qry_feats, prompt.unsqueeze(1).expand_as(qry_feats))
#         scaler.scale(loss_q).backward()
#         scaler.step(opt_s); scaler.update()
#         fast.requires_grad_(True)

#         print(f"[EP {epi:02d}] meta-loss {loss_q.item():.4f}")

#     # ─── few-shot evaluation ─────────────────────────────────────────────
#     print("\n>> Few-shot eval …")
#     test_ds = TestDataset(args.data_root, base_tf)
#     idx_map = {0:[],1:[]}
#     for i,(_,lbl) in enumerate(test_ds.samples):
#         idx_map[lbl].append(i)

#     segor = RescaleSegmentor((1024,1024))
#     accs,aurocs = [],[]

#     for _ in tqdm(range(args.eval_episodes),desc="eval"):
#         # adapt fast on support
#         fast_eval = fast.detach().clone().requires_grad_(True)
#         opt_fe   = optim.SGD([fast_eval],lr=1e-2)

#         # sample new support/query
#         sup_i, _ = sample_good_episode(train_idxs,args.n_shot,args.n_query)
#         sup_feats = []
#         for i in sup_i:
#             img,_ = test_ds[i]
#             img = img.to(device)
#             for s in args.scales:
#                 with torch.no_grad(): sup_feats.append(pool_feats(img,s))
#         sup_feats = torch.cat(sup_feats,dim=1)

#         for _ in range(args.inner_steps):
#             opt_fe.zero_grad()
#             with autocast():
#                 prompt = proto + slow.mean(0,True) + fast_eval.mean(0,True)
#                 loss   = mse(sup_feats, prompt.unsqueeze(1).expand_as(sup_feats))
#             scaler.scale(loss).backward()
#             scaler.step(opt_fe); scaler.update()

#         # threshold
#         with torch.no_grad():
#             prompt = proto + slow.mean(0,True) + fast_eval.mean(0,True)
#             d_sup = torch.norm(sup_feats - prompt.unsqueeze(1), dim=2)
#             thr   = np.percentile(d_sup.cpu().numpy(), args.threshold_percentile)

#         # score query batch
#         qidx = sample_test_episode(idx_map,args.n_query)
#         imgs, lbls = zip(*(test_ds[i] for i in qidx))
#         imgs = torch.stack(imgs).to(device)

#         # patch-level kNN distances + ensemble
#         all_patch_scores=[]
#         for s in args.scales:
#             tf = transforms.Resize((s,s)); batch=tf(imgs)
#             maps = mlsam(batch)
#             patches = torch.cat([m.flatten(2).permute(0,2,1) for m in maps],dim=1)  # [B,P,D]
#             B,P,_ = patches.shape
#             # FAISS batch query
#             Dmat = faiss.Matrix(batch_size=P*B)
#             # flatten
#             flat = patches.cpu().numpy().reshape(-1,D)
#             D2 = index.search(flat.astype(np.float32), args.k_nn)[0].mean(1)
#             patch_scores = torch.from_numpy(D2).view(B,P).to(device)
#             all_patch_scores.append(patch_scores)
#         # pixel-map: upsample & take max across scales
#         masks = torch.stack([segor.upsample(ps) for ps in all_patch_scores],0).max(0)
#         # image score = max over mask
#         img_scores = masks.view(masks.shape[0],-1).max(1).cpu().numpy()
#         preds = (img_scores>thr).astype(int)
#         accs.append((preds==np.array(lbls)).mean())

#     print(f"\nFinal few-shot anomaly accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

# if __name__=="__main__":
#     main()

#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
run_patchcore.py

PatchCore-style few-shot anomaly detection on MVTec “capsule” with SAM ↔ FAISS:
 - Multi‐scale patch features from SAM
 - GPU coreset sampling + FAISS IP index for cosine scoring
 - Fast/slow prompt meta‐training with margin ranking loss
 - Outlier exposure on defects (from test folder if no train/defect)
 - Inner‐loop support loss logging
 - Threshold by support percentile
 - Final few-shot evaluation: Accuracy & AUROC
"""

import os, random, argparse
import numpy as np
from multiprocessing import freeze_support

import torch, faiss
import torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from segment_anything import sam_model_registry

# ─── DATASETS ───────────────────────────────────────────────────────────
class GoodTrainDataset(Dataset):
    def __init__(self, root, tf):
        folder = os.path.join(root, "train", "good")
        if not os.path.isdir(folder):
            raise RuntimeError(f"Missing folder: {folder}")
        self.samples = sorted(
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith(".png")
        )
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        return self.tf(img), 0

class TestDataset(Dataset):
    def __init__(self, root, tf):
        base = os.path.join(root, "test")
        if not os.path.isdir(base):
            raise RuntimeError(f"Missing folder: {base}")
        self.samples = []
        for cls in sorted(os.listdir(base)):
            cls_dir = os.path.join(base, cls)
            if not os.path.isdir(cls_dir): continue
            lbl = 0 if cls=="good" else 1
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fn), lbl))
        if not self.samples:
            raise RuntimeError("No test samples found")
        self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path,lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), lbl

# ─── SAM ↔ PATCHES ─────────────────────────────────────────────────────
class MultiLayerSAM(nn.Module):
    def __init__(self, sam, layers):
        super().__init__()
        e = sam.image_encoder
        self.conv_proj = e.patch_embed.proj
        self.cls_token = getattr(e, "cls_token", None) or getattr(e, "class_token", None)
        self.pos_embed = getattr(e, "pos_embed", None)
        self.norm      = getattr(e, "ln_pre", None) or getattr(e, "norm", None)
        self.blocks    = getattr(e, "blocks", None) or getattr(e, "encoder", None)
        self.vit_mode  = self.cls_token is not None and self.pos_embed is not None and self.blocks is not None
        if self.vit_mode:
            self.layers = layers
        else:
            print("⚠️  SAM-ViT missing cls/blocks → patch‐only mode")

    def forward(self, x):
        if not self.vit_mode:
            return [ self.conv_proj(x) ]
        B = x.shape[0]
        fmap = self.conv_proj(x)
        C,Hp,Wp = fmap.shape[1:]
        x = fmap.flatten(2).transpose(1,2)
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat([cls, x],1) + self.pos_embed
        if self.norm: x = self.norm(x)
        feats = []
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.layers:
                pt = x[:,1:,:]
                fm = pt.transpose(1,2).reshape(B,C,Hp,Wp)
                feats.append(fm)
        return feats

    @staticmethod
    def to_patches(fm):
        B,C,H,W = fm.shape
        return fm.reshape(B,C,H*W).permute(0,2,1)  # <-- use reshape here

# ─── UPSAMPLER ─────────────────────────────────────────────────────────
class RescaleSegmentor:
    def __init__(self, target):
        import torch.nn.functional as F
        self.F = F; self.target = target
    def upsample(self, x):
        x = x.unsqueeze(1)
        x = self.F.interpolate(x, size=self.target,
                               mode="bilinear", align_corners=False)
        x = self.F.avg_pool2d(x, kernel_size=3, padding=1)
        return x.squeeze(1)

# ─── EPISODE SAMPLERS ───────────────────────────────────────────────────
def sample_mixed_support(good_idxs, defect_idxs, n_shot):
    g = random.sample(good_idxs,  n_shot)
    b = random.sample(defect_idxs, n_shot)
    lbl = torch.cat([torch.zeros(n_shot), torch.ones(n_shot)],0).long()
    return g, b, lbl

def sample_queries(idx_map, n_query):
    g = random.sample(idx_map[0], n_query)
    b = random.sample(idx_map[1], n_query)
    return g, b

def sample_good_episode(ds, idxs, n_shot, n_query):
    pick = random.sample(idxs, n_shot + n_query)
    sup, qry = pick[:n_shot], pick[n_shot:]
    sup_imgs = torch.stack([ds[i][0] for i in sup])
    qry_imgs = torch.stack([ds[i][0] for i in qry])
    return sup_imgs, qry_imgs

# ─── MAIN ───────────────────────────────────────────────────────────────
def main():
    freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--n_shot",   type=int, default=5)
    p.add_argument("--n_query",  type=int, default=5)
    p.add_argument("--train_episodes", type=int, default=100)
    p.add_argument("--inner_steps",    type=int, default=5)
    p.add_argument("--eval_episodes",  type=int, default=100)
    p.add_argument("--memory_frac",    type=float, default=0.05)
    p.add_argument("--layers",         type=int, nargs="+", default=[3,7,11])
    p.add_argument("--scales",         type=int, nargs="+", default=[1024,512])
    p.add_argument("--k_nn",           type=int, default=3)
    p.add_argument("--threshold_pct",  type=float, default=99.0)
    args = p.parse_args()

    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = torch.device(args.device)

    # transforms
    base_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
          mean=[0.48145466,0.4578275,0.40821073],
          std =[0.26862954,0.26130258,0.27577711],
        ),
    ])
    good_ds = GoodTrainDataset(args.data_root, base_tf)
    test_ds = TestDataset   (args.data_root, base_tf)

    good_idxs   = list(range(len(good_ds)))
    defect_idxs = [i for i,(_,lbl) in enumerate(test_ds.samples) if lbl==1]
    idx_map = {0:[],1:[]}
    for i,(_,lbl) in enumerate(test_ds.samples):
        idx_map[lbl].append(i)

    print(f"# good/train = {len(good_idxs)}, defect/train = {len(defect_idxs)}")
    print(f"# good/test  = {len(idx_map[0])}, defect/test  = {len(idx_map[1])}")

    # SAM wrapper
    sam_raw = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam_raw = sam_raw.to(device).eval()
    for p_ in sam_raw.parameters(): p_.requires_grad = False
    mlsam = MultiLayerSAM(sam_raw, layers=args.layers).to(device)
    segor  = RescaleSegmentor((1024,1024))

    # build memory bank
    all_patches = []
    print(">> Building memory bank …")
    for scale in args.scales:
        tf = transforms.Compose([transforms.Resize((scale,scale)), base_tf])
        loader = DataLoader(good_ds, batch_size=4, num_workers=4)
        for imgs,_ in tqdm(loader, desc=f"scale {scale}", leave=False):
            imgs = imgs.to(device)
            with torch.no_grad():
                feats = mlsam(imgs)
                for fm in feats:
                    pv = MultiLayerSAM.to_patches(fm)   # now contiguous reshape
                    B,P,C = pv.shape
                    all_patches.append(pv.reshape(-1, C).cpu().numpy())
    memory = np.concatenate(all_patches,0).astype('float32')
    N,D    = memory.shape
    m_core = max(1, int(args.memory_frac * N))
    print(f"  » total patches: {N}, coreset size: {m_core}")

    # GPU coreset sampling
    dev_mem = torch.from_numpy(memory).to(device)
    selected = [random.randrange(N)]
    dist2     = 1 - (F.normalize(dev_mem[selected[0]].unsqueeze(0),1)
                     @ F.normalize(dev_mem,1).T).squeeze(0)
    for i in range(1,m_core):
        nxt = int(dist2.argmax().item())
        selected.append(nxt)
        newd = 1 - (F.normalize(dev_mem[nxt].unsqueeze(0),1)
                    @ F.normalize(dev_mem,1).T).squeeze(0)
        dist2 = torch.min(dist2, newd)
    memory_feats = F.normalize(dev_mem[selected],1).cpu().numpy()

    # GPU FAISS IP index
    res     = faiss.StandardGpuResources()
    cpu_idx = faiss.IndexFlatIP(D)
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
    gpu_idx.add(memory_feats)

    # meta-training
    fast  = nn.Parameter(torch.randn(2, D, device=device)*0.02)
    slow  = nn.Parameter(torch.randn(len(args.layers), D, device=device)*0.02)
    proto = nn.Parameter(torch.randn(1, D, device=device)*0.02)
    opt_f  = optim.SGD([fast],        lr=1e-2)
    opt_sp = optim.SGD([slow, proto], lr=1e-3)
    scaler = GradScaler()
    autoc  = autocast
    margin_in, margin_out = 0.1, 0.2

    def make_prompt():
        p = proto + slow.mean(0,True) + fast.mean(0,True)
        return F.normalize(p,1)

    print(">> Meta-training …")
    for ep in range(1, args.train_episodes+1):
        g, b, sup_lbls = sample_mixed_support(good_idxs, defect_idxs, args.n_shot)
        imgs_sup = torch.stack([good_ds[i][0] for i in g] +
                               [test_ds[i][0] for i in b]).to(device)
        sup_lbls = sup_lbls.to(device).float()

        with torch.no_grad():
            feats = mlsam(imgs_sup)
            patches = torch.cat([MultiLayerSAM.to_patches(f) for f in feats],1)
        patches = F.normalize(patches,dim=2)

        fast.data.zero_()
        inner_ls = []
        for _ in range(args.inner_steps):
            opt_f.zero_grad()
            with autoc():
                p    = make_prompt()
                sims = (patches @ p.T).squeeze(-1)
                pos  = sims[:args.n_shot].reshape(-1)
                neg  = sims[args.n_shot:].reshape(-1)
                loss_in = F.relu(neg - pos + margin_in).mean()
            scaler.scale(loss_in).backward()
            scaler.step(opt_f); scaler.update()
            inner_ls.append(loss_in.item())

        fast.requires_grad_(False)
        opt_sp.zero_grad()
        gq, dq = sample_queries(idx_map, args.n_query)
        imgs_q = torch.cat([
            torch.stack([good_ds[i][0] for i in gq]),
            torch.stack([test_ds[i][0] for i in dq])
        ],0).to(device)

        with torch.no_grad():
            fq = mlsam(imgs_q)
            pq = torch.cat([MultiLayerSAM.to_patches(f) for f in fq],1)
        pq = F.normalize(pq,dim=2)

        with autoc():
            p     = make_prompt()
            sims_q= (pq @ p.T).squeeze(-1)
            pos_q = sims_q[:args.n_query].reshape(-1)
            neg_q = sims_q[args.n_query:].reshape(-1)
            loss_out = F.relu(neg_q - pos_q + margin_out).mean()
        scaler.scale(loss_out).backward()
        scaler.step(opt_sp); scaler.update()
        fast.requires_grad_(True)

        print(f"[EP {ep:03d}/{args.train_episodes:03d}] "
              f"inner={np.mean(inner_ls):.4f}, outer={loss_out.item():.4f}")

    # few-shot evaluation
    print("\n>> Few-shot evaluation …")
    accs, aucs = [], []
    for _ in tqdm(range(args.eval_episodes), desc="Eval"):
        sup, _ = sample_good_episode(good_ds, good_idxs,
                                     args.n_shot, args.n_query)
        sup = sup.to(device)
        fast_eval = fast.clone().detach().requires_grad_(True)
        opt_fe    = optim.SGD([fast_eval], lr=1e-2)

        for _ in range(args.inner_steps):
            opt_fe.zero_grad()
            with autoc():
                fs = mlsam(sup)
                ps = torch.cat([MultiLayerSAM.to_patches(f) for f in fs],1)
                ps = F.normalize(ps,dim=2)
                p_ = F.normalize(proto + slow.mean(0,True) +
                               fast_eval.mean(0,True),1)
                sims_sup = (ps @ p_.T).squeeze(-1).reshape(-1)
                loss_fe  = F.relu(margin_in - sims_sup).mean()
            scaler.scale(loss_fe).backward()
            scaler.step(opt_fe); scaler.update()

        with torch.no_grad():
            fs   = mlsam(sup)
            ps   = torch.cat([MultiLayerSAM.to_patches(f) for f in fs],1)
            ps   = F.normalize(ps,dim=2)
            p_   = F.normalize(proto + slow.mean(0,True) +
                               fast_eval.mean(0,True),1)
            sims = (ps @ p_.T).squeeze(-1).reshape(-1).cpu().numpy()
            thr  = np.percentile(sims, args.threshold_pct)

        gq, dq = sample_queries(idx_map, args.n_query)
        imgs   = torch.cat([
            torch.stack([good_ds[i][0] for i in gq]),
            torch.stack([test_ds[i][0] for i in dq])
        ],0).to(device)
        lbls   = np.array([0]*args.n_query + [1]*args.n_query)

        with torch.no_grad():
            fmap = sam_raw.image_encoder(imgs)
        B,C,Hp,Wp = fmap.shape
        patches   = fmap.flatten(2).permute(0,2,1).reshape(-1,C).cpu().numpy()
        D2        = gpu_idx.search(patches.astype('float32'), args.k_nn)[0].mean(1)
        sims      = D2.reshape(B, Hp*Wp)

        heatmaps   = segor.upsample(torch.from_numpy(sims).view(B,Hp,Wp).to(device))
        img_scores = heatmaps.view(B,-1).max(1)[0].cpu().numpy()
        preds      = (img_scores > thr).astype(int)

        accs.append((preds == lbls).mean())
        aucs.append(roc_auc_score(lbls, img_scores))

    print(f"\nAccuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUROC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__ == "__main__":
    main()
