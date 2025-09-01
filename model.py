"""
PatchCore Sanity Baseline (no prompts, no SAM)
=============================================
A dead-simple, *known-good* PatchCore baseline to verify your dataset, transforms,
coreset, and scoring. Use this to isolate issues. Expected: MVTec 'bottle' Image AUROC
≈ 0.98–0.99 at 224–256 when everything is wired correctly.

Run
---
  pip install torch torchvision scikit-learn pillow numpy tqdm

  python PatchCore_Sanity_Baseline.py \
    --data_root /path/to/mvtec \
    --category bottle \
    --img 224 --batch 8 --device auto \
    --per_task_cap 8000 --topk 5

Notes
-----
- No prompts, no SAM, no meta-learning. Frozen WRN50-2 features.
- Coreset via Furthest-First (FFS). L2-normalized patch embeddings.
- Image score: 99th percentile (more robust than max). Also prints max.
"""
from __future__ import annotations
import os, math, random, argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ------------------
# Dataset
# ------------------
class MVTecAD(Dataset):
    def __init__(self, root: str, category: str, split: str = "train", image_size: int = 224,
                 transform: Optional[T.Compose] = None):
        self.root = Path(root); self.category = category; self.split = split
        base = self.root / category
        assert base.exists(), f"Category not found: {base}"
        imgs: List[Path] = []; masks: List[Optional[Path]] = []; labels: List[int] = []; dtypes: List[str] = []
        if split == "train":
            for p in sorted((base / "train" / "good").glob("*.*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    imgs.append(p); masks.append(None); labels.append(0); dtypes.append("good")
        elif split == "test":
            gt_root = base / "ground_truth"
            for sub in sorted((base / "test").iterdir()):
                if not sub.is_dir(): continue
                dt = sub.name
                for p in sorted(sub.glob("*.*")):
                    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]: continue
                    imgs.append(p); dtypes.append(dt)
                    if dt == "good": masks.append(None); labels.append(0)
                    else:
                        stem = p.stem; cand = sorted((gt_root / dt).glob(f"{stem}*"))
                        masks.append(cand[0] if cand else None); labels.append(1)
        else:
            raise ValueError("split must be 'train' or 'test'")
        self.img_paths, self.mask_paths = imgs, masks
        self.labels, self.defect_types = labels, dtypes
        self.transform = transform or T.Compose([
            T.Resize(image_size, InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_resize = T.Resize(image_size, InterpolationMode.NEAREST)
        self.mask_center = T.CenterCrop(image_size)

    def __len__(self): return len(self.img_paths)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, idx: int, target_size: Tuple[int, int]) -> torch.Tensor:
        mp = self.mask_paths[idx]
        if mp is None:
            H, W = target_size
            return torch.zeros((1, H, W), dtype=torch.float32)
        m = Image.open(mp).convert("L")
        m = self.mask_resize(m); m = self.mask_center(m)
        m = torch.from_numpy((np.array(m) > 0).astype("float32"))
        return m.unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self.transform(self._load_image(self.img_paths[idx]))
        _, H, W = img.shape
        mask = self._load_mask(idx, (H, W))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"image": img, "mask": mask, "label": label,
                "path": str(self.img_paths[idx]), "defect_type": self.defect_types[idx]}

# ------------------
# Feature extractor
# ------------------
class WRNFeature(nn.Module):
    def __init__(self):
        super().__init__()
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2 = m.layer2; self.l3 = m.layer3; self.l4 = m.layer4
        for p in self.parameters(): p.requires_grad = False
        self.eval()
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f2 = self.l2(x)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        # Project each to 256 ch + common spatial via avgpool, then concat
        proj = []
        for f in (f2, f3, f4):
            c = f.shape[1]
            p = F.adaptive_avg_pool2d(f, output_size=(16,16))  # 16x16 grid
            p = F.conv2d(p, weight=torch.eye(c, device=f.device).view(c,1,1,1)[:min(c,256)], bias=None)
            if p.shape[1] > 256:
                p = p[:, :256]
            proj.append(p)
        z = torch.cat(proj, dim=1)  # (B, Csum, 16, 16)
        B, C, H, W = z.shape
        return z.permute(0,2,3,1).reshape(B, H*W, C)  # (B, N, C)

# ------------------
# Utils
# ------------------
@torch.no_grad()
def l2_normalize(x, dim=-1): return F.normalize(x, dim=dim)

@torch.no_grad()
def furthest_first(X: torch.Tensor, m: int) -> torch.Tensor:
    N = X.size(0); m = min(m, N)
    start = torch.randint(0, N, (1,), device=X.device)
    centers = [start.item()]
    dists = torch.cdist(X[start], X).squeeze(0)
    for _ in range(1, m):
        nxt = torch.argmax(dists).item(); centers.append(nxt)
        d_new = torch.cdist(X[nxt:nxt+1], X).squeeze(0)
        dists = torch.minimum(dists, d_new)
    return torch.tensor(centers, dtype=torch.long, device=X.device)

@torch.no_grad()
def score_patchcore(patch_tokens: torch.Tensor, memory_bank: torch.Tensor, topk: int = 5) -> torch.Tensor:
    B, N, C = patch_tokens.shape
    Fp = patch_tokens.reshape(B*N, C)
    K = memory_bank
    topk = min(topk, max(1, K.size(0)))
    d = torch.cdist(Fp, K)
    vals, _ = torch.topk(d, k=topk, largest=False)
    d1 = vals[:, 0]
    gap = (vals[:, -1] - vals[:, 0]).clamp(min=0)
    s = d1 * (1.0 - torch.exp(-gap))
    side = int(math.sqrt(N))
    return s.view(B, side, side)

# ------------------
# Main
# ------------------

def resolve_device(arg: str) -> torch.device:
    d = (arg or "auto").lower()
    if d in ("auto", "autodetect"):
        if torch.cuda.is_available(): return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    if d == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--category', type=str, default='bottle')
    ap.add_argument('--img', type=int, default=224)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--per_task_cap', type=int, default=8000)
    ap.add_argument('--topk', type=int, default=5)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print('[device]', device)

    # loaders
    tfm = T.Compose([
        T.Resize(args.img, InterpolationMode.BILINEAR),
        T.CenterCrop(args.img),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds_tr = MVTecAD(args.data_root, args.category, 'train', args.img, tfm)
    ds_te = MVTecAD(args.data_root, args.category, 'test',  args.img, tfm)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    ld_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    # sanity: verify train labels are all 0
    train_lbls = torch.tensor([ds_tr[i]['label'] for i in range(len(ds_tr))])
    print('[train] num=', len(ds_tr), 'unique labels=', train_lbls.unique(sorted=True))

    # model
    f = WRNFeature().to(device).eval()

    # extract train features
    feats = []
    with torch.no_grad():
        for b in tqdm(ld_tr, desc='Extract(train)'):
            x = b['image'].to(device)
            z = f(x)                         # (B,N,C)
            z = l2_normalize(z, -1).reshape(-1, z.shape[-1])
            feats.append(z.cpu())
    Fm = torch.cat(feats, dim=0)
    print('[feats] raw=', Fm.shape, 'norm=', float(F.normalize(Fm,dim=1).norm(dim=1).mean()))

    # coreset
    keep = min(args.per_task_cap, Fm.size(0))
    idx = furthest_first(F.normalize(Fm,dim=1), keep)
    bank = Fm[idx].to(device)
    print('[bank]', bank.shape)

    # evaluate
    all_img, all_lbl, all_pred_pix, all_true_pix = [], [], [], []
    with torch.no_grad():
        for b in tqdm(ld_te, desc='Evaluate'):
            x = b['image'].to(device)
            z = f(x)                         # (B,N,C)
            z = l2_normalize(z, -1)
            pm = score_patchcore(z, F.normalize(bank,dim=1), topk=args.topk)
            H, W = x.shape[-2:]
            anom_img = F.interpolate(pm[:,None], size=(H,W), mode='bilinear', align_corners=False).squeeze(1)
            img_scores_p99 = torch.quantile(anom_img.view(x.size(0), -1), q=0.99, dim=1)
            img_scores_max = anom_img.view(x.size(0), -1).max(dim=1).values
            all_img.append(img_scores_p99.cpu().numpy())
            all_lbl.append(b['label'].numpy())
            idx_def = (b['label'].numpy() == 1).nonzero()[0].reshape(-1) if np.any(b['label'].numpy()==1) else np.array([])
            if idx_def.size > 0:
                all_pred_pix.append(anom_img.cpu().numpy()[idx_def])
                all_true_pix.append(b['mask'].numpy()[idx_def])
    y = np.concatenate(all_lbl); s = np.concatenate(all_img)
    auroc = roc_auc_score(y, s)
    if all_pred_pix:
        P = np.concatenate(all_pred_pix, axis=0).reshape(-1)
        Tm = np.concatenate(all_true_pix, axis=0).reshape(-1)
        aupr = average_precision_score(Tm.astype(np.uint8), P.astype(np.float32))
    else:
        aupr = float('nan')
    print(f"Image AUROC (p99): {auroc:.4f}")

    # also print the max-score AUROC for reference
    s_max = []
    with torch.no_grad():
        for b in tqdm(ld_te, desc='Evaluate(max)'):
            x = b['image'].to(device)
            z = l2_normalize(f(x), -1)
            pm = score_patchcore(z, F.normalize(bank,dim=1), topk=args.topk)
            H, W = x.shape[-2:]
            anom_img = F.interpolate(pm[:,None], size=(H,W), mode='bilinear', align_corners=False).squeeze(1)
            s_max.append(anom_img.view(x.size(0), -1).max(dim=1).values.cpu().numpy())
    smax = np.concatenate(s_max)
    print(f"Image AUROC (max): {roc_auc_score(y, smax):.4f}")

if __name__ == '__main__':
    main()
