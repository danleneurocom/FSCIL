
"""
FSPT‑PatchCore: Brain‑Inspired Fast/Slow Prompt Tuning for Continual AD
========================================================================
A **non‑UCAD** variant that uses the mechanism from the paper
"Brain‑Inspired Fast‑ and Slow‑Update Prompt Tuning" (FSPT)
but tailored to **continual anomaly detection (CAD)** on MVTec.

Key differences vs UCAD‑style baselines
---------------------------------------
- **No keys, no per‑task prompt banks, no per‑task knowledge routing.**
- Single **global PatchCore memory** (rehearsal coreset) shared across tasks.
- **Fast/Slow visual prompts** (on a frozen ViT) trained by **meta‑learning**
  (inner loop = fast; outer loop = slow), then consolidated—exactly the
  brain‑inspired mechanism but for *visual* prompts.
- Optional **SAM‑guided SCL** to make inner/outer steps structure‑aware.

What stays the same (to remain apples‑to‑apples for AD)
------------------------------------------------------
- Feature backbone frozen; **PatchCore k‑NN reweighting** scoring.
- Memory budget via **furthest‑first sampling (FFS)**.
- Evaluate after each task on **all seen categories** using the **same current
  prompts + a single global memory** (no routing).

Usage
-----
  pip install timm torch torchvision scikit-learn pillow numpy tqdm opencv-python
  pip install git+https://github.com/facebookresearch/segment-anything.git

  # (optional) SAM checkpoint for crisper masks
  export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth

  python main.py \
    --data_root /Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d \
    --img 224 --batch 8 \
    --prompt_layers 3,4,5,6,7 --prompt_slow 10 --prompt_fast 6 \
    --inner_steps 5 --inner_lr 3e-2 --outer_lr 1e-4 \
    --mem_cap 120000 --per_task_cap 8000 --topk 5 \
    --sam_train regions --sam_test regions --sam_alpha 0.4

"""
from __future__ import annotations
import os, math, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# ------------------
# Dataset: MVTec AD
# ------------------
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class MVTecAD(Dataset):
    def __init__(self, root: str, category: str, split: str = "train", image_size: int = 224,
                 transform: Optional[T.Compose] = None):
        self.root = Path(root); self.category = category; self.split = split; self.image_size = image_size
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
                    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                        continue
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
            T.CenterCrop(image_size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_resize = T.Resize(image_size, InterpolationMode.NEAREST)
        self.mask_center = T.CenterCrop(image_size)

    def __len__(self): return len(self.img_paths)
    def _load_image(self, p: Path) -> Image.Image: return Image.open(p).convert("RGB")
    def _load_mask(self, idx: int, hw: Tuple[int, int]) -> torch.Tensor:
        mp = self.mask_paths[idx]
        if mp is None:
            H, W = hw; return torch.zeros((1, H, W), dtype=torch.float32)
        m = Image.open(mp).convert("L"); m = self.mask_resize(m); m = self.mask_center(m)
        m = torch.from_numpy((np.array(m) > 0).astype("float32"))
        return m.unsqueeze(0)
    def __getitem__(self, idx: int) -> Dict[str, object]:
        img = self.transform(self._load_image(self.img_paths[idx]))
        _, H, W = img.shape
        mask = self._load_mask(idx, (H, W))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"image": img, "mask": mask, "label": label,
                "path": str(self.img_paths[idx]), "defect_type": self.defect_types[idx]}


def load_mvtec_category(root: str, category: str, image_size: int = 224, batch_size: int = 8,
                        num_workers: int = 0, pin_memory: bool = False):
    ds_tr = MVTecAD(root, category, split="train", image_size=image_size)
    ds_te = MVTecAD(root, category, split="test",  image_size=image_size)
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                       pin_memory=pin_memory, persistent_workers=num_workers > 0)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return ds_tr, ds_te, ld_tr, ld_te

# ----------------------------
# Backbone: ViT (timm)
# ----------------------------
import timm
from timm.models.vision_transformer import VisionTransformer

class PromptedViT(nn.Module):
    """Visual fast/slow prompts injected at chosen blocks, removed after each block.
    ViT weights are **frozen**; only prompt params update (fast in inner loop,
    slow in outer loop), mirroring FSPT's mechanism.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224.augreg_in21k",
                 prompt_layers: List[int] = [3,4,5,6,7],
                 slow_len: int = 10, fast_len: int = 6,
                 prompt_init_std: float = 0.02,
                 mid_layer_index: int = 5):
        super().__init__()
        self.vit: VisionTransformer = timm.create_model(model_name, pretrained=True)
        assert isinstance(self.vit, VisionTransformer)
        for p in self.vit.parameters(): p.requires_grad = False
        self.vit.eval()
        self.D = self.vit.embed_dim
        self.prompt_layers = sorted(set(prompt_layers))
        self.slow_len, self.fast_len = slow_len, fast_len
        self.mid_layer_index = mid_layer_index
        # slow prompts (global/meta‑knowledge)
        self.slow_prompts = nn.ParameterDict(); self.slow_pos = nn.ParameterDict()
        for li in self.prompt_layers:
            self.slow_prompts[str(li)] = nn.Parameter(torch.randn(1, slow_len, self.D) * prompt_init_std)
            pe = nn.Parameter(torch.zeros(1, slow_len, self.D)); nn.init.trunc_normal_(pe, std=prompt_init_std)
            self.slow_pos[str(li)] = pe
        # fast prompts (adapt quickly each session)
        self.fast_prompts = nn.ParameterDict(); self.fast_pos = nn.ParameterDict()
        for li in self.prompt_layers:
            self.fast_prompts[str(li)] = nn.Parameter(torch.randn(1, fast_len, self.D) * prompt_init_std)
            pe = nn.Parameter(torch.zeros(1, fast_len, self.D)); nn.init.trunc_normal_(pe, std=prompt_init_std)
            self.fast_pos[str(li)] = pe
        # init fast prompts to zero for stability
        with torch.no_grad():
            for k in self.fast_prompts: self.fast_prompts[k].zero_(); self.fast_pos[k].zero_()

    # toggles
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False
    def unfreeze_fast(self):
        for p in self.parameters(): p.requires_grad = False
        for d in (self.fast_prompts, self.fast_pos):
            for p in d.parameters(): p.requires_grad = True
    def unfreeze_slow(self):
        for p in self.parameters(): p.requires_grad = False
        for d in (self.slow_prompts, self.slow_pos):
            for p in d.parameters(): p.requires_grad = True

    @torch.no_grad()
    def forward_no_prompt_mid(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit; B = x.size(0)
        x = vit.patch_embed(x)
        cls = vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + vit.pos_embed; x = vit.pos_drop(x)
        mid = None
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i == self.mid_layer_index:
                mid = x[:, 1:, :].clone()
        x = vit.norm(x)
        return mid  # (B,N,D)

    def forward_with_prompts(self, x: torch.Tensor, return_mid: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        vit = self.vit; B = x.size(0)
        x = vit.patch_embed(x)
        x = torch.cat([vit.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + vit.pos_embed; x = vit.pos_drop(x)
        mid_tokens = None
        for i, blk in enumerate(vit.blocks):
            if i in self.prompt_layers:
                sp = self.slow_prompts[str(i)].expand(B, -1, -1) + self.slow_pos[str(i)]
                fp = self.fast_prompts[str(i)].expand(B, -1, -1) + self.fast_pos[str(i)]
                x = torch.cat([x[:, :1, :], sp, fp, x[:, 1:, :]], dim=1)
                x = blk(x)
                # remove prompts to keep token length (1+N)
                x = torch.cat([x[:, :1, :], x[:, 1 + self.slow_len + self.fast_len:, :]], dim=1)
            else:
                x = blk(x)
            if return_mid and i == self.mid_layer_index:
                mid_tokens = x[:, 1:, :].clone()
        x = vit.norm(x)
        return x[:, 1:, :], mid_tokens

# ----------------------------
# SAM mask provider
# ----------------------------
class MaskProvider:
    def __init__(self, mode: str = "none"):
        self.mode = mode
        self.sam = None; self.predictor = None
        if mode != "none":
            try:
                from segment_anything import sam_model_registry, SamPredictor
                ckpt = os.environ.get("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
                self.sam = sam_model_registry["vit_b"](checkpoint=ckpt)
                self.predictor = SamPredictor(self.sam)
                self.sam.eval()
            except Exception as e:
                print(f"[MaskProvider] SAM unavailable ({e}); falling back to 'none'.")
                self.mode = "none"

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> List[np.ndarray]:
        B, C, H, W = images.shape
        if self.mode == "none":
            return [np.ones((H, W), dtype=np.int32) for _ in range(B)]
        masks: List[np.ndarray] = []
        for b in range(B):
            x = images[b].detach().cpu()
            x = x * torch.tensor(IMAGENET_STD).view(3,1,1) + torch.tensor(IMAGENET_MEAN).view(3,1,1)
            x = (x.clamp(0,1).permute(1,2,0).numpy() * 255).astype(np.uint8)
            self.predictor.set_image(x)
            Hh, Ww = x.shape[:2]
            box = np.array([0, 0, Ww-1, Hh-1])
            m, _, _ = self.predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=True)
            m0 = m[0].astype(np.uint8)
            try:
                import cv2
                num, cc = cv2.connectedComponents(m0)
                cc = cc.astype(np.int32)
            except Exception:
                cc = m0.astype(np.int32)
            cc = (cc > 0).astype(np.int32)
            masks.append(cc)
        return masks

# ----------------------------
# Loss: SAM‑guided Supervised Contrastive
# ----------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__(); self.t = temperature
    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels >= 0
        feats = F.normalize(feats[mask], dim=-1)
        labels = labels[mask]
        if feats.size(0) <= 1: return feats.sum() * 0
        sim = feats @ feats.t() / self.t
        sim = sim - sim.max(dim=1, keepdim=True).values
        exp_sim = torch.exp(sim)
        same = labels.view(-1,1) == labels.view(1,-1)
        eye = torch.eye(same.size(0), dtype=torch.bool, device=same.device)
        pos = same & (~eye)
        denom = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim))
        denom = torch.clamp(denom, min=1e-8)
        log_prob = sim - torch.log(denom.unsqueeze(1))
        loss = -(log_prob[pos]).mean() if pos.any() else feats.sum()*0
        return loss

# ----------------------------
# PatchCore memory & scoring (global)
# ----------------------------
@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1): return F.normalize(x, dim=dim)

@torch.no_grad()
def furthest_first(X: torch.Tensor, m: int) -> torch.Tensor:
    N = X.size(0); m = int(min(m, N))
    if m <= 0: return torch.empty(0, dtype=torch.long, device=X.device)
    start = torch.randint(0, N, (1,), device=X.device)
    centers = [start.item()]
    dists = torch.cdist(X[start], X).squeeze(0)
    for _ in range(1, m):
        nxt = torch.argmax(dists).item(); centers.append(nxt)
        d_new = torch.cdist(X[nxt:nxt+1], X).squeeze(0)
        dists = torch.minimum(dists, d_new)
    return torch.tensor(centers, dtype=torch.long, device=X.device)

class GlobalMemory:
    def __init__(self, mem_cap: int = 120_000):
        self.mem_cap = mem_cap
        self.bank = torch.empty(0, 1)
    @torch.no_grad()
    def add(self, feats: torch.Tensor):
        # feats: (N,C), assumed L2‑normalized, on current device
        if self.bank.numel() == 0:
            self.bank = feats.detach().clone()
        else:
            self.bank = torch.cat([self.bank, feats.detach().clone()], dim=0)
        if self.bank.size(0) > self.mem_cap:
            idx = furthest_first(self.bank, self.mem_cap)
            self.bank = self.bank[idx]
    def get(self) -> torch.Tensor:
        if self.bank.numel() == 0:
            return torch.empty(0, 1)
        return self.bank

@torch.no_grad()
def score_patchcore(patch_tokens: torch.Tensor, memory_bank: torch.Tensor, topk: int = 5, chunk: int = 4096) -> torch.Tensor:
    # patch_tokens: (B,N,C) L2‑normed; memory: (M,C) L2‑normed
    B, N, C = patch_tokens.shape
    Fp = patch_tokens.reshape(B*N, C)
    K = memory_bank
    topk = min(topk, K.size(0))
    vals_all = []
    for i in range(0, Fp.size(0), chunk):
        sl = slice(i, min(i+chunk, Fp.size(0)))
        d = torch.cdist(Fp[sl], K)
        vals, _ = torch.topk(d, k=topk, largest=False)
        vals_all.append(vals)
    vals = torch.cat(vals_all, dim=0)
    d1 = vals[:, 0]
    gap = (vals[:, -1] - vals[:, 0]).clamp(min=0)
    s = d1 * (1.0 - torch.exp(-gap))
    return s.view(B, int(math.sqrt(N)), int(math.sqrt(N)))

# ----------------------------
# Model wrapper
# ----------------------------
@dataclass
class TrainCfg:
    img: int = 224
    batch: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    prompt_layers: List[int] = field(default_factory=lambda: [3,4,5,6,7])
    slow_len: int = 10
    fast_len: int = 6
    mid_index: int = 5
    inner_steps: int = 5
    inner_lr: float = 3e-2
    outer_lr: float = 1e-4
    mem_cap: int = 120_000
    per_task_cap: int = 8_000  # per‑task addition before global cap
    topk: int = 5
    chunk: int = 4096
    sam_train: str = "regions"   # 'none'|'foreground'|'regions'
    sam_test: str = "regions"
    sam_alpha: float = 0.4

class FSPT_PatchCore(nn.Module):
    def __init__(self, cfg: TrainCfg):
        super().__init__(); self.cfg = cfg
        self.device0 = torch.device(cfg.device)
        self.vitp = PromptedViT(prompt_layers=cfg.prompt_layers, slow_len=cfg.slow_len,
                                fast_len=cfg.fast_len, mid_layer_index=cfg.mid_index)
        self.masker_train = MaskProvider(cfg.sam_train)
        self.masker_test = MaskProvider(cfg.sam_test)
        self.crit = SupConLoss(temperature=0.2)
        self.memory = GlobalMemory(mem_cap=cfg.mem_cap)
        self.to(self.device0)

    # ---- helpers ----
    @torch.no_grad()
    def _extract_mid_with_prompts(self, loader: DataLoader) -> torch.Tensor:
        feats = []
        for b in tqdm(loader, desc="Extract(with‑prompts)"):
            x = b["image"].to(self.device0)
            _, mid = self.vitp.forward_with_prompts(x)
            feats.append(mid.reshape(-1, mid.shape[-1]))
        Fm = torch.cat(feats, dim=0)
        return l2_normalize(Fm, -1)

    # ---- training (per task) ----
    def train_task(self, train_loader: DataLoader):
        cfg = self.cfg
        ds = train_loader.dataset
        # split 50/50 support/query
        idxs = list(range(len(ds))); random.Random(cfg.seed).shuffle(idxs)
        split = max(1, len(ds)//2)
        sup = Subset(ds, idxs[:split]); qry = Subset(ds, idxs[split:])
        sup_loader = DataLoader(sup, batch_size=cfg.batch, shuffle=True)
        qry_loader = DataLoader(qry, batch_size=cfg.batch, shuffle=True)

        # (1) inner loop: fast prompts
        self.vitp.unfreeze_fast(); self.train()
        opt_fast = torch.optim.SGD([p for p in self.vitp.parameters() if p.requires_grad], lr=cfg.inner_lr, momentum=0.9)
        for step in range(cfg.inner_steps):
            losses = []
            for b in tqdm(sup_loader, desc=f"Inner fast s{step+1}/{cfg.inner_steps}"):
                x = b["image"].to(self.device0)
                with torch.no_grad(): rmask = self.masker_train(x)
                _, mid = self.vitp.forward_with_prompts(x)
                B, N, D = mid.shape; side = int(math.sqrt(N))
                labels = []
                for bi in range(B):
                    rm = torch.tensor(rmask[bi], device=self.device0)
                    rm_small = F.interpolate(rm[None, None].float(), size=(side, side), mode="nearest").squeeze().long()
                    labels.append(rm_small.reshape(-1))
                labels = torch.stack(labels, dim=0).reshape(-1)
                feats = mid.reshape(-1, D)
                loss = self.crit(feats, labels)
                opt_fast.zero_grad(set_to_none=True); loss.backward(); opt_fast.step(); losses.append(loss.item())
            print(f"[FAST] loss={np.mean(losses):.4f}")

        # (2) outer loop: slow prompts
        self.vitp.unfreeze_slow(); self.train()
        opt_slow = torch.optim.AdamW([p for p in self.vitp.parameters() if p.requires_grad], lr=cfg.outer_lr, weight_decay=1e-4)
        losses = []
        for b in tqdm(qry_loader, desc="Outer slow"):
            x = b["image"].to(self.device0)
            with torch.no_grad(): rmask = self.masker_train(x)
            _, mid = self.vitp.forward_with_prompts(x)
            B, N, D = mid.shape; side = int(math.sqrt(N))
            labels = []
            for bi in range(B):
                rm = torch.tensor(rmask[bi], device=self.device0)
                rm_small = F.interpolate(rm[None, None].float(), size=(side, side), mode="nearest").squeeze().long()
                labels.append(rm_small.reshape(-1))
            labels = torch.stack(labels, dim=0).reshape(-1)
            feats = mid.reshape(-1, D)
            loss = self.crit(feats, labels)
            opt_slow.zero_grad(set_to_none=True); loss.backward(); opt_slow.step(); losses.append(loss.item())
        print(f"[SLOW] loss={np.mean(losses) if losses else 0.0:.4f}")

        # (3) update global memory with current task normals
        with torch.no_grad():
            # coreset per task before merging to global
            feats = self._extract_mid_with_prompts(train_loader)  # (N,C) L2‑normed
            keep = min(cfg.per_task_cap, feats.size(0))
            if keep < feats.size(0):
                idx = furthest_first(feats, keep)
                feats = feats[idx]
            self.memory.add(feats.to(self.device0))
        self.vitp.freeze_all(); self.eval()

    # ---- inference ----
    @torch.no_grad()
    def score_images(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        _, mid = self.vitp.forward_with_prompts(images.to(self.device0))
        mid = l2_normalize(mid, -1)
        mem = l2_normalize(self.memory.get().to(self.device0), -1)
        pm = score_patchcore(mid, mem, topk=self.cfg.topk, chunk=self.cfg.chunk)
        # optional SAM smoothing at test time
        if self.cfg.sam_test in ("foreground", "regions"):
            masks = self.masker_test(images.to(self.device0))
            side = pm.shape[-1]
            maps_ref = []
            for bi in range(images.size(0)):
                rm = torch.tensor(masks[bi], device=self.device0)
                rm_small = F.interpolate(rm[None, None].float(), size=(side, side), mode="nearest").squeeze()
                if self.cfg.sam_test == "foreground":
                    m = pm[bi] * (rm_small > 0).float()
                else:
                    reg_ids = rm_small.long().view(-1); pm_flat = pm[bi].view(-1)
                    out = torch.zeros_like(pm_flat)
                    for r in reg_ids.unique():
                        idx = (reg_ids == r).nonzero(as_tuple=False).view(-1)
                        if idx.numel() == 0: continue
                        mean_r = pm_flat[idx].mean()
                        out[idx] = self.cfg.sam_alpha * mean_r + (1 - self.cfg.sam_alpha) * pm_flat[idx]
                    m = out.view(side, side)
                maps_ref.append(m)
            pm = torch.stack(maps_ref, dim=0)
        H, W = images.shape[-2:]
        anom_img = F.interpolate(pm[:, None], size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        img_scores = anom_img.view(images.size(0), -1).max(dim=1).values
        return img_scores.detach().cpu().numpy(), anom_img.detach().cpu().numpy()

# ----------------------------
# Evaluation loop
# ----------------------------
@torch.no_grad()
def evaluate(model: FSPT_PatchCore, loader: DataLoader) -> Tuple[float, float]:
    all_img_scores, all_labels = [], []
    all_pred_maps, all_true_masks = [], []
    for b in tqdm(loader, desc="Evaluate"):
        imgs = b["image"]
        labels = b["label"].numpy()
        s_img, s_map = model.score_images(imgs)
        all_img_scores.append(s_img); all_labels.append(labels)
        idx = np.where(labels == 1)[0]
        if idx.size > 0:
            all_pred_maps.append(s_map[idx]); all_true_masks.append(b["mask"].numpy()[idx])
    y = np.concatenate(all_labels); s = np.concatenate(all_img_scores)
    img_auroc = roc_auc_score(y, s)
    if all_pred_maps:
        P = np.concatenate(all_pred_maps, axis=0).reshape(-1).astype(np.float32)
        Tm = np.concatenate(all_true_masks, axis=0).reshape(-1).astype(np.uint8)
        pix_aupr = average_precision_score(Tm, P)
    else:
        pix_aupr = float('nan')
    return float(img_auroc), float(pix_aupr)


def forgetting_measure(history: Dict[str, List[float]]) -> float:
    vals = []
    for _, seq in history.items():
        if not seq: continue
        vals.append(max(seq) - seq[-1])
    return float(np.mean(vals)) if vals else 0.0

@dataclass
class RunResult:
    per_task_traj: Dict[str, List[float]] = field(default_factory=dict)
    per_task_last: Dict[str, Dict[str, float]] = field(default_factory=dict)
    FM: float = 0.0

# ----------------------------
# Orchestrator
# ----------------------------

def run_sequence(data_root: str, categories: List[str], cfg: TrainCfg) -> RunResult:
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    model = FSPT_PatchCore(cfg).to(cfg.device)
    traj: Dict[str, List[float]] = {c: [] for c in categories}

    for i, cat in enumerate(categories, 1):
        print(f"\n=== Task {i}/{len(categories)}: {cat} ===")
        tr, te, ld_tr, ld_te = load_mvtec_category(data_root, cat, image_size=cfg.img, batch_size=cfg.batch)
        model.train_task(ld_tr)
        print(f"[Memory] size={model.memory.get().shape[0]}")
        for eval_cat in categories[:i]:
            _, _, _, te_loader = load_mvtec_category(data_root, eval_cat, image_size=cfg.img, batch_size=cfg.batch)
            img_auc, pix_aupr = evaluate(model, te_loader)
            traj[eval_cat].append(img_auc)
            print(f"  [after {cat:>12}] {eval_cat:12}  Image AUROC={img_auc:.4f}  Pixel AUPR={(np.nan if np.isnan(pix_aupr) else pix_aupr):.4f}")

    fm = forgetting_measure(traj)
    last = {c: {"final_img_auroc": seq[-1]} for c, seq in traj.items() if seq}
    print("\n=== Summary ===")
    for c, seq in traj.items():
        hist = ", ".join(f"t{j}:{v:.4f}" for j, v in enumerate(seq, 1))
        print(f"{c:12} -> {hist}")
    print(f"FM (avg max minus final): {fm:.4f}")
    return RunResult(per_task_traj=traj, per_task_last=last, FM=fm)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--tasks", type=str, default="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper")
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--prompt_layers", type=str, default="3,4,5,6,7")
    p.add_argument("--prompt_slow", type=int, default=10)
    p.add_argument("--prompt_fast", type=int, default=6)
    p.add_argument("--inner_steps", type=int, default=5)
    p.add_argument("--inner_lr", type=float, default=3e-2)
    p.add_argument("--outer_lr", type=float, default=1e-4)
    p.add_argument("--mem_cap", type=int, default=120000)
    p.add_argument("--per_task_cap", type=int, default=8000)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--sam_train", type=str, default="regions", choices=["none","foreground","regions"])
    p.add_argument("--sam_test", type=str, default="regions", choices=["none","foreground","regions"])
    p.add_argument("--sam_alpha", type=float, default=0.4)
    args = p.parse_args()

    cfg = TrainCfg(
        img=args.img, batch=args.batch, device=args.device, seed=args.seed,
        prompt_layers=[int(x) for x in args.prompt_layers.split(',') if x],
        slow_len=args.prompt_slow, fast_len=args.prompt_fast, mid_index=5,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr, outer_lr=args.outer_lr,
        mem_cap=args.mem_cap, per_task_cap=args.per_task_cap, topk=args.topk, chunk=args.chunk,
        sam_train=args.sam_train, sam_test=args.sam_test, sam_alpha=args.sam_alpha,
    )

    cats = args.tasks.split()
    _ = run_sequence(args.data_root, cats, cfg)
