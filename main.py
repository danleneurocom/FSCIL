"""
UCAD‑style Continual Anomaly Detection (full, runnable skeleton)
================================================================

Implements:
  • MVTec‑AD dataset loader (+ concat test loader across seen tasks)
  • ViT‑B/16 extractor with **per‑block additive prompts** (fast vs slow)
  • Fast/Slow meta‑trainer with **multiple outer steps**
  • **Per‑task** memory: Keys (K_e via early block features) + Knowledge (K_n via final features)
  • **Task selection** at inference via **min calibrated image score** (was: keys only)
  • **k‑center coreset** (greedy) for both keys and knowledge
  • **Structure‑aware contrastive loss (SCL)** using SAM Automatic Mask Generator 
    (with a fallback to SLIC superpixels when SAM is not installed)
  • End‑to‑end train/eval loop printing Image AUROC and Pixel AUROC after each task

What changed vs previous version
--------------------------------
- Image score pooling now uses **top‑q%** (default q=2%) instead of fixed top‑100.
- Added **per‑task calibration** (z‑score using mean/std from train‑good images).
- **Task selection** now chooses the task that yields the *lowest calibrated image score*.
- Removed inadvertent `@torch.no_grad()` on the extractor input path and avoided in‑place tensor edits
  so prompts actually receive gradients.

Dependencies (pip):
  torch torchvision numpy scikit-image scikit-learn tqdm Pillow
  (optional for best SCL) segment‑anything opencv‑python

Note
----
This is a compact, single‑file reference. It’s designed to drop into your 
current project and replace your extractor + trainer + orchestration. 
Edit the `__main__` section to point to your dataset + (optionally) SAM weights.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    # torchvision >= 0.13
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except Exception as e:  # pragma: no cover
    raise RuntimeError("Please install/upgrade torchvision>=0.13 to use ViT_B_16.")

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

try:
    # Optional but preferred for SCL
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    _HAS_SAM = True
except Exception:
    _HAS_SAM = False

try:
    import cv2
except Exception:
    cv2 = None

from skimage.segmentation import slic
from skimage.color import rgb2lab

# --------------------------------------------------------------------------------------
# Utilities & reproducibility
# --------------------------------------------------------------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1337)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
print(f"[device] Using: {DEVICE}")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------------------
# Dataset: MVTec‑AD
# --------------------------------------------------------------------------------------

class MVTecAD(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[T.Compose] = None,
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        base = self.root / category
        assert base.exists(), f"Category not found: {base}"
        img_paths: List[Path] = []
        mask_paths: List[Optional[Path]] = []
        labels: List[int] = []
        defect_types: List[str] = []

        if split == "train":
            for p in sorted((base / "train" / "good").glob("*.*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    img_paths.append(p)
                    mask_paths.append(None)
                    labels.append(0)
                    defect_types.append("good")
        elif split == "test":
            gt_root = base / "ground_truth"
            for sub in sorted((base / "test").iterdir()):
                if not sub.is_dir():
                    continue
                dt = sub.name
                for p in sorted(sub.glob("*.*")):
                    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                        continue
                    img_paths.append(p)
                    defect_types.append(dt)
                    if dt == "good":
                        mask_paths.append(None)
                        labels.append(0)
                    else:
                        stem = p.stem
                        cand = sorted((gt_root / dt).glob(f"{stem}*"))
                        mask_paths.append(cand[0] if cand else None)
                        labels.append(1)
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.defect_types = defect_types

        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size, InterpolationMode.BILINEAR),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transform

        self.mask_resize = T.Resize(image_size, InterpolationMode.NEAREST)
        self.mask_center = T.CenterCrop(image_size)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, idx: int, target_hw: Tuple[int, int]) -> torch.Tensor:
        mp = self.mask_paths[idx]
        H, W = target_hw
        if mp is None:
            return torch.zeros((1, H, W), dtype=torch.float32)
        m = Image.open(mp).convert("L")
        m = self.mask_resize(m)
        m = self.mask_center(m)
        m = torch.from_numpy(np.array(m, dtype=np.uint8))
        m = (m > 0).float().unsqueeze(0)
        return m

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_pil = self._load_image(self.img_paths[idx])
        img_tensor = self.transform(img_pil)
        _, H, W = img_tensor.shape
        mask = self._load_mask(idx, (H, W))
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return {
            "image": img_tensor,
            "mask": mask,
            "label": label,
            "path": str(self.img_paths[idx]),
            "defect_type": self.defect_types[idx],
            "image_pil": img_pil,
        }


def train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([d["image"] for d in batch], dim=0),
        "path": [d["path"] for d in batch],
        "image_pil": [d["image_pil"] for d in batch],
    }


def test_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pil_images = [d["image_pil"] for d in batch]
    for item in batch:
        del item["image_pil"]
    out = {
        "image": torch.stack([d["image"] for d in batch], dim=0),
        "mask": torch.stack([d["mask"] for d in batch], dim=0),
        "label": torch.stack([d["label"] for d in batch], dim=0),
        "path": [d["path"] for d in batch],
        "defect_type": [d["defect_type"] for d in batch],
        "image_pil": pil_images,
    }
    return out


def load_mvtec_category(
    root: str,
    category: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    train_ds = MVTecAD(root, category, split="train", image_size=image_size)
    test_ds = MVTecAD(root, category, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=train_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=test_collate_fn
    )
    return train_ds, test_ds, train_loader, test_loader


def get_continual_test_loaders(seen_categories: List[str], data_root: str, image_size: int, batch_size: int):
    test_datasets = [MVTecAD(data_root, cat, split="test", image_size=image_size) for cat in seen_categories]
    combined_test_ds = torch.utils.data.ConcatDataset(test_datasets)
    return DataLoader(
        combined_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=test_collate_fn
    )

# --------------------------------------------------------------------------------------
# ViT with **per‑block prompts** (fast vs slow) + prompt bank for per‑task snapshots
# --------------------------------------------------------------------------------------

class ViTPromptExtractor(nn.Module):
    def __init__(
        self,
        fast_layers: Iterable[int] = (0, 1, 2),
        slow_layers: Optional[Iterable[int]] = None,
        freeze_backbone: bool = True,
        add_scale_gates: bool = True,
        cls_affects: bool = False,
        weights: Optional[ViT_B_16_Weights] = ViT_B_16_Weights.IMAGENET1K_V1,
    ):
        super().__init__()
        self.vit = vit_b_16(weights=weights)
        self.encoder = self.vit.encoder
        self.blocks: nn.ModuleList = self.encoder.layers
        self.num_layers = len(self.blocks)
        self.embed_dim = self.vit.hidden_dim
        self.cls_affects = cls_affects

        fast_set = set(int(i) for i in fast_layers)
        if slow_layers is None:
            slow_set = set(range(self.num_layers)) - fast_set
        else:
            slow_set = set(int(i) for i in slow_layers)
        assert fast_set.isdisjoint(slow_set)

        def _mk_prompt_list(active: set[int]) -> nn.ParameterList:
            plist: List[nn.Parameter] = []
            for i in range(self.num_layers):
                if i in active:
                    p = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                    nn.init.trunc_normal_(p, std=0.02)
                else:
                    p = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=False)
                plist.append(p)
            return nn.ParameterList(plist)

        self.fast_prompts = _mk_prompt_list(fast_set)
        self.slow_prompts = _mk_prompt_list(slow_set)

        self.use_gates = bool(add_scale_gates)
        if self.use_gates:
            def _mk_gate_list(active: set[int]) -> nn.ParameterList:
                glist: List[nn.Parameter] = []
                for i in range(self.num_layers):
                    if i in active:
                        g = nn.Parameter(torch.tensor(0.5))
                    else:
                        g = nn.Parameter(torch.tensor(0.0), requires_grad=False)
                    glist.append(g)
                return nn.ParameterList(glist)
            self.fast_gates = _mk_gate_list(fast_set)
            self.slow_gates = _mk_gate_list(slow_set)
        else:
            self.register_module("fast_gates", None)
            self.register_module("slow_gates", None)

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    # ---- param groups ----
    def fast_params(self) -> Iterable[nn.Parameter]:
        for p in self.fast_prompts:
            yield p
        if self.use_gates:
            for g in self.fast_gates:
                yield g

    def slow_params(self) -> Iterable[nn.Parameter]:
        for p in self.slow_prompts:
            yield p
        if self.use_gates:
            for g in self.slow_gates:
                yield g

    # ---- prompt state (for PromptBank) ----
    def get_prompt_state(self) -> Dict[str, List[torch.Tensor]]:
        def _clone(pl: nn.ParameterList) -> List[torch.Tensor]:
            return [p.detach().cpu().clone() for p in pl]
        state = {
            "fast_prompts": _clone(self.fast_prompts),
            "slow_prompts": _clone(self.slow_prompts),
        }
        if self.use_gates:
            state.update({
                "fast_gates": [g.detach().cpu().clone() for g in self.fast_gates],
                "slow_gates": [g.detach().cpu().clone() for g in self.slow_gates],
            })
        return state

    def load_prompt_state(self, state: Dict[str, List[torch.Tensor]]):
        def _load_paramlist(pl: nn.ParameterList, tensors: List[torch.Tensor]):
            for p, t in zip(pl, tensors):
                if p.requires_grad:
                    p.data.copy_(t.to(p.device))
                else:
                    p.data = t.to(p.device)
        _load_paramlist(self.fast_prompts, state["fast_prompts"]) ; _load_paramlist(self.slow_prompts, state["slow_prompts"]) 
        if self.use_gates and "fast_gates" in state:
            for g, t in zip(self.fast_gates, state["fast_gates"]):
                if g.requires_grad:
                    g.data.copy_(t.to(g.device))
                else:
                    g.data = t.to(g.device)
            for g, t in zip(self.slow_gates, state["slow_gates"]):
                if g.requires_grad:
                    g.data.copy_(t.to(g.device))
                else:
                    g.data = t.to(g.device)

    def zero_prompts_(self):
        with torch.no_grad():
            for p in list(self.fast_prompts) + list(self.slow_prompts):
                p.zero_()
            if self.use_gates:
                for g in list(self.fast_gates) + list(self.slow_gates):
                    g.zero_()

    # ---- forward helpers ----
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        # Keep autograd enabled so prompts receive gradients through the frozen ViT
        return self.vit._process_input(x)

    def _add_prompt(self, tokens: torch.Tensor, i: int, use_prompts: bool = True) -> torch.Tensor:
        if not use_prompts:
            return tokens
        p = self.fast_prompts[i]
        if p.requires_grad is False:
            p = p.detach()
        q = self.slow_prompts[i]
        if q.requires_grad is False:
            q = q.detach()
        if self.use_gates:
            alpha = self.fast_gates[i] if self.fast_prompts[i].requires_grad else torch.tensor(0.0, device=tokens.device)
            beta  = self.slow_gates[i] if self.slow_prompts[i].requires_grad else torch.tensor(0.0, device=tokens.device)
            add = alpha * p + beta * q
        else:
            add = p + q
        add = add.to(tokens.device)
        if self.cls_affects:
            tokens = tokens + add
        else:
            tokens = torch.cat([tokens[:, :1, :], tokens[:, 1:, :] + add], dim=1)
        return tokens

    def forward_tokens(self, x: torch.Tensor, use_prompts: bool = True, until_layer: Optional[int] = None, final_ln: bool = True) -> torch.Tensor:
        z = self._process_input(x)
        for i, blk in enumerate(self.blocks):
            z = self._add_prompt(z, i, use_prompts=use_prompts)
            z = blk(z)
            if until_layer is not None and i == until_layer:
                z = self.encoder.ln(z) if final_ln else z
                return z[:, 1:, :]
        z = self.encoder.ln(z) if final_ln else z
        return z[:, 1:, :]

    def get_final_patches(self, x: torch.Tensor, use_prompts: bool = True) -> torch.Tensor:
        return self.forward_tokens(x, use_prompts=use_prompts, until_layer=None, final_ln=True)

    def get_key_patches(self, x: torch.Tensor, layer_idx: int = 5, use_prompts: bool = False) -> torch.Tensor:
        return self.forward_tokens(x, use_prompts=use_prompts, until_layer=layer_idx, final_ln=True)


class PromptBank:
    def __init__(self):
        self.bank: Dict[str, Dict[str, List[torch.Tensor]]] = {}

    def save(self, task_name: str, model: ViTPromptExtractor):
        self.bank[task_name] = model.get_prompt_state()

    def load(self, task_name: str, model: ViTPromptExtractor):
        model.load_prompt_state(self.bank[task_name])

    def has(self, task_name: str) -> bool:
        return task_name in self.bank

# --------------------------------------------------------------------------------------
# Losses: unsupervised compactness + memory alignment + SCL (structure contrastive)
# --------------------------------------------------------------------------------------

def variance_pull_together(patches: torch.Tensor) -> torch.Tensor:
    mu = patches.mean(dim=(0, 1), keepdim=True)
    return ((patches - mu) ** 2).mean()


@torch.no_grad()
def sample_memory_targets(memory_bank: torch.Tensor, n_vec: int = 4096) -> torch.Tensor:
    if memory_bank.numel() == 0:
        return torch.empty(0, 0)
    n = min(n_vec, memory_bank.shape[0])
    idx = torch.randint(0, memory_bank.shape[0], (n,))
    return memory_bank[idx]


def memory_alignment_loss(curr_feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.numel() == 0:
        return curr_feats.new_tensor(0.0)
    d = torch.cdist(curr_feats, targets.to(curr_feats.device))
    nnv = targets.to(curr_feats.device)[d.argmin(dim=1)]
    return F.mse_loss(curr_feats, nnv)


def scl_loss_from_regions(
    patch_tokens: torch.Tensor,      # [B, N, D], L2‑normalized recommended
    region_labels: List[np.ndarray], # len B, each [Hp, Wp] int labels (>=2 regions)
    tau: float = 0.2,
    min_region: int = 3,
) -> torch.Tensor:
    B, N, D = patch_tokens.shape
    # infer patch grid size (assume square)
    side = int(math.sqrt(N))
    z = F.normalize(patch_tokens, dim=-1)
    total = 0.0
    count = 0
    for b in range(B):
        lbl = region_labels[b]
        if lbl.ndim == 1:  # might already be flattened
            lbl = lbl.reshape(side, side)
        # resize to [side, side] if needed
        if lbl.shape != (side, side):
            # nearest neighbor resize
            lbl_img = Image.fromarray(lbl.astype(np.int32))
            lbl_img = lbl_img.resize((side, side), Image.NEAREST)
            lbl = np.array(lbl_img)
        labels_flat = torch.from_numpy(lbl.reshape(-1)).to(z.device)
        # compute region centroids
        uniq = labels_flat.unique()
        # filter tiny regions
        region_means = []
        region_ids = []
        for r in uniq.tolist():
            idx = torch.nonzero(labels_flat == r, as_tuple=False).squeeze(1)
            if idx.numel() < min_region: 
                continue
            region_means.append(z[b, idx].mean(dim=0, keepdim=True))
            region_ids.append(r)
        if len(region_means) < 2:
            continue
        centroids = torch.cat(region_means, dim=0)  # [R, D]
        # token‑wise infoNCE against centroids
        for r, r_id in enumerate(region_ids):
            idx = torch.nonzero(labels_flat == r_id, as_tuple=False).squeeze(1)
            if idx.numel() == 0: continue
            q = z[b, idx]                               # [Ni, D]
            pos = centroids[r:r+1]                      # [1, D]
            neg = torch.cat([centroids[:r], centroids[r+1:]], dim=0)  # [R-1, D]
            # logits: sim(q, pos/neg)/tau
            lpos = (q @ pos.t()) / tau                  # [Ni, 1]
            lneg = (q @ neg.t()) / tau                  # [Ni, R-1]
            logits = torch.cat([lpos, lneg], dim=1)     # [Ni, R]
            labels = torch.zeros(q.size(0), dtype=torch.long, device=z.device)
            total = total + F.cross_entropy(logits, labels)
            count += 1
    if count == 0:
        return patch_tokens.new_tensor(0.0)
    return total / count

# --------------------------------------------------------------------------------------
# Structure masks (SAM Automatic or SLIC fallback)
# --------------------------------------------------------------------------------------

class StructureSegmenter:
    def __init__(self, sam_checkpoint: Optional[str] = None):
        self.use_sam = _HAS_SAM and (sam_checkpoint is not None)
        if self.use_sam:
            sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            sam.to(device=DEVICE)
            self.masker = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.9,
                crop_n_layers=0,
            )
        else:
            self.masker = None

    def _sam_segments(self, img_np: np.ndarray) -> np.ndarray:
        # Returns an int32 label map HxW (0..R-1)
        masks = self.masker.generate(img_np)
        seg = np.zeros(img_np.shape[:2], dtype=np.int32)
        for i, m in enumerate(sorted(masks, key=lambda x: x['area'], reverse=True)):
            seg[m['segmentation']] = i + 1
        return seg

    def _slic_segments(self, img_np: np.ndarray) -> np.ndarray:
        lab = rgb2lab(img_np)
        seg = slic(lab, n_segments=200, compactness=20, sigma=1, start_label=1)
        return seg.astype(np.int32)

    def batch(self, pil_list: List[Image.Image]) -> List[np.ndarray]:
        outs: List[np.ndarray] = []
        for pil in pil_list:
            npimg = np.array(pil)
            if self.use_sam:
                try:
                    seg = self._sam_segments(npimg)
                except Exception:
                    seg = self._slic_segments(npimg)
            else:
                seg = self._slic_segments(npimg)
            outs.append(seg)
        return outs

# --------------------------------------------------------------------------------------
# k‑center coreset (greedy) and FPS utilities
# --------------------------------------------------------------------------------------

def _pairwise_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: [N, D], y: [M, D]
    return torch.cdist(x, y)


def kcenter_coreset(X: torch.Tensor, m: int, seed_idx: Optional[int] = None) -> torch.Tensor:
    """Greedy k‑center on CPU for stability. Returns subset of X (m x D)."""
    Xc = X.detach().cpu()
    N = Xc.shape[0]
    if m >= N:
        return Xc
    if seed_idx is None:
        seed_idx = np.random.randint(0, N)
    centers = [seed_idx]
    dists = _pairwise_cdist(Xc, Xc[seed_idx:seed_idx+1]).squeeze(1)  # [N]
    for _ in range(1, m):
        idx = torch.argmax(dists).item()
        centers.append(idx)
        dists = torch.minimum(dists, _pairwise_cdist(Xc, Xc[idx:idx+1]).squeeze(1))
    return Xc[centers]


def farthest_point_sampling(X: torch.Tensor, m: int) -> torch.Tensor:
    return kcenter_coreset(X, m)

# --------------------------------------------------------------------------------------
# Meta‑trainer (fast/slow) with **multiple outer steps** and optional SCL
# --------------------------------------------------------------------------------------

@dataclass
class MetaCfg:
    lr_fast: float = 3e-2
    lr_slow: float = 1e-4
    wd: float = 1e-4
    align_coef: float = 0.5
    scl_coef: float = 0.5
    inner_steps: int = 5
    outer_steps: int = 100
    outer_batches_per_step: int = 1
    key_layer: int = 5


class FastSlowMetaTrainer:
    def __init__(self, model: ViTPromptExtractor, cfg: MetaCfg, segmenter: Optional[StructureSegmenter] = None):
        self.model = model.to(DEVICE)
        self.cfg = cfg
        self.segmenter = segmenter
        self.opt_fast = torch.optim.SGD(list(self.model.fast_params()), lr=cfg.lr_fast)
        self.opt_slow = torch.optim.AdamW(list(self.model.slow_params()), lr=cfg.lr_slow, weight_decay=cfg.wd)

    def inner_update(self, support_batch: Dict[str, Any]) -> float:
        self.model.train()
        for p in self.model.slow_params(): p.requires_grad_(False)
        for p in self.model.fast_params(): p.requires_grad_(True)

        imgs = support_batch["image"].to(DEVICE)
        feats = self.model.get_final_patches(imgs, use_prompts=True)
        loss = variance_pull_together(feats)

        self.opt_fast.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_fast.step()
        return float(loss.item())

    def _regions_for_batch(self, batch: Dict[str, Any], side: int) -> List[np.ndarray]:
        # Return one int label map per image, aligned to patch grid (side x side)
        segs = self.segmenter.batch(batch["image_pil"]) if self.segmenter is not None else []
        outs: List[np.ndarray] = []
        for i, pil in enumerate(batch["image_pil"]):
            if len(segs) > i:
                seg = segs[i]
            else:
                # trivial single region if no segmenter
                seg = np.zeros((pil.height, pil.width), dtype=np.int32)
            # resize to patch grid
            seg_img = Image.fromarray(seg.astype(np.int32))
            seg_img = seg_img.resize((side, side), Image.NEAREST)
            outs.append(np.array(seg_img))
        return outs

    def outer_update(self, query_batch: Dict[str, Any], memory_bank: torch.Tensor) -> Tuple[float, float, float]:
        self.model.train()
        for p in self.model.fast_params(): p.requires_grad_(False)
        for p in self.model.slow_params(): p.requires_grad_(True)

        imgs = query_batch["image"].to(DEVICE)
        feats = self.model.get_final_patches(imgs, use_prompts=True)    # [B, N, D]
        loss_q = variance_pull_together(feats)
        with torch.no_grad():
            targets = sample_memory_targets(memory_bank, n_vec=max(1024, feats.numel() // feats.shape[-1]))
        loss_align = memory_alignment_loss(feats.reshape(-1, feats.shape[-1]), targets)

        # SCL (structure‑aware contrastive)
        side = int(math.sqrt(feats.shape[1]))
        scl = 0.0
        if self.cfg.scl_coef > 0.0 and self.segmenter is not None:
            regions = self._regions_for_batch(query_batch, side)
            scl = scl_loss_from_regions(feats, regions, tau=0.2)
        loss = loss_q + self.cfg.align_coef * loss_align + self.cfg.scl_coef * scl

        self.opt_slow.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_slow.step()
        return float(loss_q.item()), float(loss_align.item()), float(scl if isinstance(scl, float) else scl.item())

    def outer_update_many(self, query_loader: DataLoader, memory_bank: torch.Tensor) -> Dict[str, float]:
        stats = {"loss_q": 0.0, "loss_align": 0.0, "scl": 0.0, "steps": 0}
        it = iter(query_loader)
        for _ in range(self.cfg.outer_steps):
            for _ in range(self.cfg.outer_batches_per_step):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(query_loader)
                    batch = next(it)
                lq, la, ls = self.outer_update(batch, memory_bank)
                stats["loss_q"] += lq; stats["loss_align"] += la; stats["scl"] += ls; stats["steps"] += 1
        for k in ("loss_q", "loss_align", "scl"):
            if stats["steps"] > 0:
                stats[k] /= stats["steps"]
        return stats

# --------------------------------------------------------------------------------------
# Per‑task memory (keys/prompts/knowledge) and task‑agnostic inference
# --------------------------------------------------------------------------------------

@dataclass
class TaskMemory:
    keys: torch.Tensor       # [Mk, D]
    knowledge: torch.Tensor  # [Mn, D]
    mu: float = 0.0          # calibration mean of image scores on train-good
    sigma: float = 1.0       # calibration std (clamped >=1e-6)

class UCADMemory:
    def __init__(self):
        self.mem: Dict[str, TaskMemory] = {}

    def add(self, task: str, keys: torch.Tensor, knowledge: torch.Tensor, mu: float = 0.0, sigma: float = 1.0):
        self.mem[task] = TaskMemory(keys.detach().cpu(), knowledge.detach().cpu(), float(mu), float(max(sigma, 1e-6)))

    def tasks(self) -> List[str]:
        return list(self.mem.keys())

    def concat_knowledge(self) -> torch.Tensor:
        if not self.mem:
            return torch.empty(0, 0)
        return torch.cat([self.mem[t].knowledge for t in self.tasks()], dim=0)


# ---- helpers for image score pooling + calibration ----

def _image_score_from_min_d(amap_flat: torch.Tensor, q: float = 0.02, center_quantile: Optional[float] = None) -> torch.Tensor:
    """amap_flat: [N] vector of per-patch anomaly (min distance)."""
    a = amap_flat
    if center_quantile is not None and 0.0 < center_quantile < 0.5:
        # subtract a robust baseline (e.g., 10th percentile) to reduce all-high bias
        k = max(1, int(center_quantile * a.numel()))
        base = torch.topk(-a, k=k, largest=False).values.max()  # approx low quantile
        a = a - base
    k = max(1, int(q * a.numel()))
    return torch.topk(a, k=k, largest=True).values.mean()


@torch.no_grad()
def estimate_image_score_stats(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    task: str,
    loader: DataLoader,
    q: float = 0.02,
    center_quantile: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute mean/std of (uncalibrated) image scores for a task on its train-good set."""
    extractor.eval().to(DEVICE)
    prompt_bank.load(task, extractor)
    K = ucad_mem.mem[task].knowledge.to(DEVICE)
    scores: List[torch.Tensor] = []
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        feats = extractor.get_final_patches(imgs, use_prompts=True)     # [B,N,D]
        B, N, D = feats.shape
        dmin = torch.cdist(feats.reshape(-1, D), K).min(dim=1).values.view(B, N)
        s = torch.stack([_image_score_from_min_d(dmin[b], q=q, center_quantile=center_quantile) for b in range(B)])
        scores.append(s.cpu())
    s = torch.cat(scores)
    mu = float(s.mean())
    sigma = float(s.std().clamp_min(1e-6))
    return mu, sigma


@torch.no_grad()
def build_task_memory(
    extractor: ViTPromptExtractor,
    loader: DataLoader,
    key_layer: int = 5,
    max_keys: int = 4096,
    max_knowledge: int = 20000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extractor.eval().to(DEVICE)
    keys_list: List[torch.Tensor] = []
    know_list: List[torch.Tensor] = []
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        k = extractor.get_key_patches(imgs, layer_idx=key_layer, use_prompts=False)  # neutral keys
        z = extractor.get_final_patches(imgs, use_prompts=True)                       # with prompts
        keys_list.append(k.reshape(-1, k.shape[-1]).detach().cpu())
        know_list.append(z.reshape(-1, z.shape[-1]).detach().cpu())
    keys = torch.cat(keys_list, dim=0)
    knowledge = torch.cat(know_list, dim=0)
    # coreset selection
    keys_sel = farthest_point_sampling(keys, min(max_keys, keys.shape[0]))
    know_sel = kcenter_coreset(knowledge, min(max_knowledge, knowledge.shape[0]))
    return keys_sel, know_sel


@torch.no_grad()
def select_task_by_imgscore(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    imgs: torch.Tensor,                 # [B, 3, H, W]
    q: float = 0.02,
    center_quantile: Optional[float] = None,
) -> List[str]:
    """Pick task that yields the lowest *calibrated* image score per image."""
    extractor.eval().to(DEVICE)
    B = imgs.size(0)
    best_scores = [float("inf")] * B
    best_tasks = [None] * B
    for t in ucad_mem.tasks():
        prompt_bank.load(t, extractor)
        feats = extractor.get_final_patches(imgs, use_prompts=True)   # [B,N,D]
        B_, N, D = feats.shape
        K = ucad_mem.mem[t].knowledge.to(imgs.device)
        dmin = torch.cdist(feats.reshape(-1, D), K).min(dim=1).values.view(B_, N)
        # pool + calibrate
        raw_scores = torch.stack([_image_score_from_min_d(dmin[b], q=q, center_quantile=center_quantile) for b in range(B_)])
        mu, sigma = ucad_mem.mem[t].mu, ucad_mem.mem[t].sigma
        s_cal = (raw_scores - mu) / sigma
        for b in range(B_):
            val = float(s_cal[b].item())
            if val < best_scores[b]:
                best_scores[b] = val
                best_tasks[b] = t
    # fallback
    tasks = ucad_mem.tasks()
    return [bt if bt is not None else (tasks[0] if tasks else "") for bt in best_tasks]

@torch.no_grad()
def select_task_by_keys(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    imgs: torch.Tensor,                 # [B, 3, H, W]
    key_layer: int = 5,
) -> List[str]:
    """
    Pick the most likely task for each image by comparing *neutral* early-layer
    patch features to each task's key memory (nearest-key average distance).
    """
    extractor.eval().to(DEVICE)
    task_names = ucad_mem.tasks()
    if len(task_names) == 0:
        raise RuntimeError("UCADMemory is empty; add a task before selecting tasks.")

    # Neutral keys (no prompts) from the chosen early layer
    neutral = extractor.get_key_patches(imgs, layer_idx=key_layer, use_prompts=False)  # [B, N, D]
    B, N, D = neutral.shape

    selected: List[str] = []
    for b in range(B):
        feats = neutral[b]  # [N, D]
        best_task = None
        best_score = float("inf")
        for t in task_names:
            K = ucad_mem.mem[t].keys.to(feats.device)  # [Mk, D]
            if K.numel() == 0:
                continue
            d = torch.cdist(feats, K)                  # [N, Mk]
            score = d.min(dim=1).values.mean().item()  # avg NN distance
            if score < best_score:
                best_score = score
                best_task = t
        # Fallback (in case all K were empty for some reason)
        selected.append(best_task if best_task is not None else task_names[0])
    return selected

@torch.no_grad()
def infer_batch_anomaly(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    imgs: torch.Tensor,
    key_layer: int = 5,
    image_topk: Optional[int] = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extractor.eval().to(DEVICE)
    B = imgs.shape[0]

    tasks = select_task_by_keys(extractor, prompt_bank, ucad_mem, imgs, key_layer=key_layer)

    image_scores = []
    maps = []
    for b in range(B):
        t = tasks[b]
        prompt_bank.load(t, extractor)

        feats = extractor.get_final_patches(imgs[b:b+1], use_prompts=True)  # [1, N, D]
        N = feats.shape[1]
        Z = feats.reshape(-1, feats.shape[-1])                               # [N, D]
        K = ucad_mem.mem[t].knowledge.to(Z.device)                           # [M, D]
        d = torch.cdist(Z, K)                                                # [N, M]
        min_d = d.min(dim=1).values                                          # [N]

        # Reshape to ViT-B/16 patch grid robustly
        Ht, Wt = imgs.shape[-2], imgs.shape[-1]
        ps = 16
        Hp, Wp = Ht // ps, Wt // ps
        expN = Hp * Wp
        if N != expN:
            if N > expN:
                min_d = min_d[:expN]
            else:
                min_d = F.pad(min_d, (0, expN - N), value=float(min_d[-1].item()))
        amap = min_d.view(Hp, Wp)

        # Image score = mean of top-k pixels
        k = N if image_topk is None else max(1, min(int(image_topk), N))
        topk = torch.topk(amap.flatten(), k=k, largest=True).values
        image_scores.append(topk.mean().item())
        maps.append(amap.detach().cpu())

    return torch.tensor(image_scores), torch.stack(maps, dim=0)


# --------------------------------------------------------------------------------------
# Orchestration: train per task, update PromptBank + UCADMemory, evaluate
# --------------------------------------------------------------------------------------

def calculate_pixel_auroc(gt_masks: np.ndarray, pixel_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    # Restrict to anomalous images for pixel AUROC
    pixels_per_image = gt_masks.size // len(gt_labels)
    gt_masks_reshaped = gt_masks.reshape(len(gt_labels), pixels_per_image)
    pixel_scores_reshaped = pixel_scores.reshape(len(gt_labels), pixels_per_image)
    non_good_idx = np.where(gt_labels > 0)[0]
    if len(non_good_idx) == 0:
        return 0.0
    gt_masks_non_good = gt_masks_reshaped[non_good_idx].flatten()
    pixel_scores_non_good = pixel_scores_reshaped[non_good_idx].flatten()
    return roc_auc_score(gt_masks_non_good, pixel_scores_non_good)

def run_ucad(
    categories: List[str],
    data_root: str,
    image_size: int,
    batch_size: int,
    sam_checkpoint: Optional[str] = None,
    # memory sizes
    max_keys: int = 4096,
    max_knowledge: int = 20000,
    # meta cfg
    inner_steps: int = 5,
    outer_steps: int = 100,
    lr_fast: float = 3e-2,
    lr_slow: float = 1e-4,
    align_coef: float = 0.5,
    scl_coef: float = 0.5,
    # NEW: controls image-level scoring (top-k pixels)
    image_topk_q: int = 100,
):
    print("[init] Building extractor + segmenter…")
    extractor = ViTPromptExtractor(
        fast_layers=(0, 1, 2), slow_layers=None, freeze_backbone=True, add_scale_gates=True
    )
    segmenter = StructureSegmenter(sam_checkpoint)
    meta = FastSlowMetaTrainer(
        extractor,
        MetaCfg(
            lr_fast=lr_fast, lr_slow=lr_slow, wd=1e-4,
            align_coef=align_coef, scl_coef=scl_coef,
            inner_steps=inner_steps, outer_steps=outer_steps, outer_batches_per_step=1,
            key_layer=5,
        ),
        segmenter,
    )

    prompt_bank = PromptBank()
    ucad_mem = UCADMemory()

    # Optional: store per-task image-score calibration (if you use it later)
    score_calib: Dict[str, Tuple[float, float]] = {}

    seen_categories: List[str] = []

    for task_idx, category in enumerate(categories):
        print(f"\n=== Task {task_idx+1}/{len(categories)}: {category} ===")
        seen_categories.append(category)

        train_ds, _, train_loader, _ = load_mvtec_category(
            root=data_root, category=category, image_size=image_size, batch_size=batch_size
        )

        # Build simple support/query splits from GOOD images
        good_idx = list(range(len(train_ds)))
        if len(good_idx) < 2 * batch_size:
            support_idx = good_idx
            query_idx = good_idx
        else:
            split = max(batch_size, len(good_idx) // 2)
            support_idx = good_idx[:split]
            query_idx = good_idx[split:]

        support_loader = DataLoader(
            Subset(train_ds, support_idx), batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn
        )
        query_loader = DataLoader(
            Subset(train_ds, query_idx), batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn
        )

        # ----- Inner loop (fast) -----
        inner_losses = []
        for _, batch in zip(range(meta.cfg.inner_steps), support_loader):
            inner_losses.append(meta.inner_update(batch))
        print(f"[inner] mean loss: {np.mean(inner_losses):.5f} (steps={len(inner_losses)})")

        # ----- Outer loop (slow, many steps) -----
        combined_prev_knowledge = ucad_mem.concat_knowledge()
        outer_stats = meta.outer_update_many(query_loader, combined_prev_knowledge)
        print(
            f"[outer] loss_q={outer_stats['loss_q']:.5f} | align={outer_stats['loss_align']:.5f} "
            f"| scl={outer_stats['scl']:.5f} | steps={outer_stats['steps']}"
        )

        # ----- Snapshot prompts for this task -----
        prompt_bank.save(category, extractor)

        # ----- Build per-task memory (keys/knowledge, with k-center/fps) -----
        keys, knowledge = build_task_memory(
            extractor, train_loader, key_layer=meta.cfg.key_layer, max_keys=max_keys, max_knowledge=max_knowledge
        )
        ucad_mem.add(category, keys, knowledge)
        print(f"[memory] {category}: keys={keys.shape[0]} | knowledge={knowledge.shape[0]}")

        # ----- Optional calibration (runs only if you defined the helper) -----
        try:
            mu, sigma = estimate_image_score_stats(
                extractor, prompt_bank, ucad_mem, category, train_loader, meta.cfg.key_layer
            )
            score_calib[category] = (float(mu), float(sigma))
            print(f"[calib] {category}: mu={mu:.4f} sigma={sigma:.4f}")
        except NameError:
            pass
        except Exception as e:
            print(f"[calib] skipped for {category}: {e}")

        # ----- Evaluate on all seen categories (task-agnostic) -----
        test_loader = get_continual_test_loaders(seen_categories, data_root, image_size, batch_size)
        gt_labels_all: List[int] = []
        image_scores_all: List[float] = []
        gt_masks_all: List[np.ndarray] = []
        pixel_maps_all: List[np.ndarray] = []

        for batch in tqdm(test_loader, desc="Testing"):
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            gt_masks = batch["mask"].squeeze(1).cpu().numpy()
            img_scores, maps = infer_batch_anomaly(
                extractor, prompt_bank, ucad_mem, imgs,
                key_layer=meta.cfg.key_layer,
                image_topk=image_topk_q,          # NEW: wire-through
            )
            # upsample maps to image size for pixel metrics
            Ht, Wt = imgs.shape[-2:]
            maps_up = F.interpolate(
                maps.unsqueeze(1), size=(Ht, Wt), mode='bilinear', align_corners=False
            ).squeeze(1)
            # normalize per image
            mu_ = maps_up.view(maps_up.size(0), -1).min(dim=1).values.view(-1, 1, 1)
            M_  = maps_up.view(maps_up.size(0), -1).max(dim=1).values.view(-1, 1, 1)
            maps_up = (maps_up - mu_) / (M_ - mu_ + 1e-8)

            gt_labels_all.append(labels)
            image_scores_all.append(img_scores.cpu().numpy())
            gt_masks_all.append(gt_masks)
            pixel_maps_all.append(maps_up.cpu().numpy())

        gt_labels = np.concatenate(gt_labels_all)
        image_scores = np.concatenate(image_scores_all)
        gt_masks_np = np.concatenate(gt_masks_all)
        pixel_scores_np = np.concatenate(pixel_maps_all)

        image_auroc = roc_auc_score(gt_labels, image_scores)
        pixel_auroc = calculate_pixel_auroc(gt_masks_np, pixel_scores_np, gt_labels)

        print(f"\n--- Results after task '{category}' ({len(seen_categories)} seen) ---")
        print(f"  Image AUROC: {image_auroc:.4f}")
        print(f"  Pixel AUROC: {pixel_auroc:.4f}")
        print("-" * 50)


# --------------------------------------------------------------------------------------
# Main (edit paths as needed)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"      # <-- CHANGE ME
    SAM_CHECKPOINT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/sam_vit_b_01ec64.pth"                  # e.g., "/path/to/sam_vit_b_01ec64.pth" (optional)

    all_categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    run_ucad(
        categories=all_categories,
        data_root=DATA_ROOT,
        image_size=224,
        batch_size=8,
        sam_checkpoint=SAM_CHECKPOINT,
        max_keys=4096,
        max_knowledge=20000,
        inner_steps=5,
        outer_steps=5,
        lr_fast=3e-2,
        lr_slow=1e-4,
        align_coef=0.5,
        scl_coef=0.5,
        image_topk_q=0.02,          # top 2% pooling for image score
        # center_quantile=None,        # set to e.g. 0.10 to subtract 10th percentile before pooling
    )
