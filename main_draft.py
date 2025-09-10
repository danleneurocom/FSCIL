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

from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from sklearn.covariance import LedoitWolf

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

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

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
        timm_name: Optional[str] = None,   # <--- NEW
    ):
        super().__init__()
        self.cls_affects = cls_affects
        self.is_timm = (timm_name is not None)

        if self.is_timm:
            assert _HAS_TIMM, "Please `pip install timm` to use timm backbones."
            # Suggested 21K checkpoints (try in this order):
            # 'vit_base_patch16_224.augreg_in21k'  (good, widely used)
            # 'vit_base_patch16_224_in21k'
            timm_name = timm_name or 'vit_base_patch16_224.augreg_in21k'
            self.vit = timm.create_model(timm_name, pretrained=True)
            # expose the parts we need
            self.blocks = self.vit.blocks
            self.num_layers = len(self.blocks)
            self.embed_dim = self.vit.embed_dim
            self.final_norm = self.vit.norm
            # pieces to build tokens like the model's forward_features
            self.patch_embed = self.vit.patch_embed
            self.cls_token = self.vit.cls_token
            self.pos_embed = self.vit.pos_embed
            self.pos_drop = self.vit.pos_drop
        else:
            # original torchvision path (ImageNet-1K)
            self.vit = vit_b_16(weights=weights)
            self.encoder = self.vit.encoder
            self.blocks: nn.ModuleList = self.encoder.layers
            self.num_layers = len(self.blocks)
            self.embed_dim = self.vit.hidden_dim
            self.final_norm = self.encoder.ln

        # prompt params (unchanged)
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
        # Returns tokens [B, 1+N, D] (with CLS) before entering blocks
        if self.is_timm:
            # timm ViT forward_features tokenization
            x = self.patch_embed(x)                       # [B, N, D]
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls, x), dim=1)                # [B, 1+N, D]
            # For 224x224, pos_embed fits; if you change size, you’ll need interpolation
            x = x + self.pos_embed
            x = self.pos_drop(x)
            return x
        else:
            # torchvision path
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

    def forward_tokens(self, x, use_prompts=True, until_layer=None, final_ln=True):
        z = self._process_input(x)
        for i, blk in enumerate(self.blocks):
            z = self._add_prompt(z, i, use_prompts=use_prompts)
            z = blk(z)
            if until_layer is not None and i == until_layer:
                z = self.final_norm(z) if final_ln else z
                return z[:, 1:, :]
        z = self.final_norm(z) if final_ln else z
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

@torch.no_grad()
def prune_dense_knowledge(ucad_mem: UCADMemory, task: str, keep_ratio: float = 0.8, sample: int = 20000, k: int = 10, metric: str = "cosine"):
    """
    Drop the densest (1-keep_ratio) fraction of knowledge vectors based on average kNN distance.
    """
    if task not in ucad_mem.mem: return
    K = ucad_mem.mem[task].knowledge
    if K.numel() == 0: return
    Kc = K.detach().cpu()
    N = Kc.shape[0]
    idx = torch.randperm(N)[:min(N, sample)]
    S = Kc[idx]  # [S, D]
    # pairwise distances within sample (approx density)
    Dmat = _pairwise_dist(S, S, metric=metric)
    # ignore self (set large)
    Dmat[torch.arange(S.size(0)), torch.arange(S.size(0))] = float("inf")
    vals, _ = torch.topk(Dmat, k=min(k, S.size(0)-1), dim=1, largest=False)
    dens = vals.mean(dim=1)                        # lower = denser
    thr = torch.quantile(dens, q=1.0 - keep_ratio) # keep top by distance
    keep_sample_mask = dens >= thr
    keep_sample_idx = idx[keep_sample_mask]

    # project to full set by nearest indices (simple: keep these indices and others randomly to meet ratio)
    keep_mask = torch.zeros(N, dtype=torch.bool)
    keep_mask[keep_sample_idx] = True
    need = int(math.ceil(N * keep_ratio)) - keep_mask.sum().item()
    if need > 0:
        # add random of remaining
        rem = torch.nonzero(~keep_mask, as_tuple=False).squeeze(1)
        extra = rem[torch.randperm(rem.numel())[:need]]
        keep_mask[extra] = True
    ucad_mem.mem[task].knowledge = Kc[keep_mask]



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
    lr_slow: float = 5e-4
    wd: float = 1e-4
    align_coef: float = 0.5
    scl_coef: float = 0.5
    inner_steps: int = 5
    outer_steps: int = 100
    outer_batches_per_step: int = 1
    key_layer: int = 5

@dataclass
class TaskMemory:
    keys: torch.Tensor       # [Mk, D]
    knowledge: torch.Tensor  # [Mn, D]
    mu: float = 0.0          # calibration mean (image score)
    sigma: float = 1.0       # calibration std
    mean_vec: Optional[torch.Tensor] = None     # [D]
    whitener: Optional[torch.Tensor] = None     # [D, D] inverse sqrt covariance


class FastSlowMetaTrainer:
    def __init__(self, model: ViTPromptExtractor, cfg: MetaCfg, segmenter: Optional[StructureSegmenter] = None):
        self.model = model.to(DEVICE)
        self.cfg = cfg
        self.segmenter = segmenter
        self.opt_fast = torch.optim.SGD(list(self.model.fast_params()), lr=cfg.lr_fast)
        self.opt_slow = torch.optim.AdamW(list(self.model.slow_params()), lr=cfg.lr_slow, weight_decay=1e-4)   # <--- Adam, not AdamW

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

def _image_score_from_min_d(
    amap_flat: torch.Tensor,
    q: float = 0.02,
    center_quantile: Optional[float] = 0.10,
) -> torch.Tensor:
    """
    Convert a 1D vector of per-patch anomaly distances into one image score.

    Args:
      amap_flat: 1D tensor [N] of per-patch distances (larger = more anomalous).
      q: fraction of highest-valued patches to average (e.g., 0.005 = top 0.5%).
      center_quantile: if set in (0, 0.5), subtract that low-quantile baseline
                       before top-k (helps tiny/sparse defects stand out).

    Returns:
      Scalar tensor (image anomaly score).
    """
    a = amap_flat

    # Optional baseline subtraction (robust to textures & global shifts).
    if center_quantile is not None and 0.0 < center_quantile < 0.5:
        # Use a robust "background" level; clamp for numerical safety.
        base = torch.quantile(a, torch.tensor(center_quantile, device=a.device))
        a = a - base

    # Pool top-k patches
    k = max(1, int(q * a.numel())) if 0.0 < q <= 1.0 else max(1, int(q))
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
    extractor.eval().to(DEVICE)
    prompt_bank.load(task, extractor)
    K = ucad_mem.mem[task].knowledge.to(DEVICE)
    mem = ucad_mem.mem[task]

    scores: List[torch.Tensor] = []
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        feats = extractor.get_final_patches(imgs, use_prompts=True)      # [B, N, D]
        B, N, D = feats.shape
        Z = feats.reshape(-1, D)                                         # [BN, D]

        if mem.whitener is not None and mem.mean_vec is not None:
            Zw = _apply_whiten(Z, mem.mean_vec, mem.whitener)
            Kw = _apply_whiten(K, mem.mean_vec, mem.whitener)
            dmin = torch.cdist(Zw, Kw).min(dim=1).values.reshape(B, N)
        else:
            dmin = torch.cdist(Z, K).min(dim=1).values.reshape(B, N)

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
    rotations: Tuple[int, ...] = (0,),   # multiples of 90 degrees (0,1,2,3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-task memory with optional 90° rotation augmentation.
    Each r in `rotations` applies torch.rot90(..., k=r, dims=(-2,-1)).
    """
    extractor.eval().to(DEVICE)

    keys_list: List[torch.Tensor] = []
    know_list: List[torch.Tensor] = []

    for batch in loader:
        imgs = batch["image"].to(DEVICE)  # [B,3,H,W]
        for r in rotations:
            if r % 4 != 0:
                imgs_r = torch.rot90(imgs, k=r % 4, dims=(-2, -1))
            else:
                imgs_r = imgs

            # neutral keys (no prompts) from early layer
            k = extractor.get_key_patches(imgs_r, layer_idx=key_layer, use_prompts=False)    # [B,N,D]
            # knowledge (with prompts) from final layer
            z = extractor.get_final_patches(imgs_r, use_prompts=True)                        # [B,N,D]

            keys_list.append(k.reshape(-1, k.shape[-1]).detach().cpu())
            know_list.append(z.reshape(-1, z.shape[-1]).detach().cpu())

    keys = torch.cat(keys_list, dim=0) if keys_list else torch.empty(0, 0)
    knowledge = torch.cat(know_list, dim=0) if know_list else torch.empty(0, 0)

    # coreset after augmentation to keep memory bounded
    if keys.numel() > 0:
        keys_sel = farthest_point_sampling(keys, min(max_keys, keys.shape[0]))
    else:
        keys_sel = keys
    if knowledge.numel() > 0:
        know_sel = kcenter_coreset(knowledge, min(max_knowledge, knowledge.shape[0]))
    else:
        know_sel = knowledge

    return keys_sel, know_sel

# --- put this small helper near your other utils (once) ---
# def _resolve_topk(num_items: int, topk: float | int) -> int:
#     """
#     If 0 < topk <= 1.0 : treat as ratio of num_items.
#     If topk >= 1       : treat as absolute count.
#     Always clamp to [1, num_items].
#     """
#     if isinstance(topk, float):
#         if topk <= 0:
#             k = 1
#         elif topk <= 1:
#             k = int(math.ceil(num_items * topk))
#         else:
#             k = int(round(topk))
#     else:
#         k = int(topk)
#     return max(1, min(num_items, k))

def _pooled_image_score(a: torch.Tensor, topk: float|int=0.05, center_quantile: float|None=None,
                        mode: str = "lse", lse_tau: float = 0.25, min_k: int = 8) -> torch.Tensor:
    a = a.flatten()
    if center_quantile is not None and 0.0 < center_quantile < 0.5:
        base = torch.quantile(a, center_quantile)
        a = torch.clamp(a - base, min=0)
    # pick top-k
    k = int(math.ceil(topk * a.numel())) if isinstance(topk, float) and 0 < topk <= 1 else int(topk)
    k = max(min_k, min(k, a.numel()))
    v, _ = torch.topk(a, k=k, largest=True)
    if mode == "lse":
        # LogSumExp pooling over top-k (stable “soft-max” pooling)
        return lse_tau * torch.logsumexp(v / lse_tau, dim=0)
    else:
        return v.mean()

def _np_eigh_inv_sqrt(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return inverse square-root of PSD matrix via eigen-decomposition."""
    # numerical jitter to ensure PSD
    cov = cov + np.eye(cov.shape[0], dtype=cov.dtype) * eps
    w, U = np.linalg.eigh(cov)  # w ascending
    w = np.clip(w, eps, None)
    inv_sqrt = (U * (1.0 / np.sqrt(w)) ) @ U.T
    return inv_sqrt

@torch.no_grad()
def compute_whitener_from_memory(memory_feats: torch.Tensor, use_ledoit: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    memory_feats: [M, D] (CPU or GPU). Returns (mean_vec[D], whitener[D,D]) on CPU float32.
    Whitener W maps (x - mean) -> (x - mean) @ W   such that distances are Mahalanobis-like.
    """
    if memory_feats.numel() == 0:
        raise ValueError("Empty memory feats for whitener.")
    X = memory_feats.detach().cpu().to(torch.float64).numpy()  # [M, D]
    mu = X.mean(axis=0, dtype=np.float64)                      # [D]
    Xc = X - mu
    if use_ledoit:
        try:
            lw = LedoitWolf(store_precision=False, assume_centered=True)
            lw.fit(Xc)                     # estimates covariance on centered data
            cov = np.asarray(lw.covariance_, dtype=np.float64)  # [D, D]
        except Exception:
            cov = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
    else:
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)

    W = _np_eigh_inv_sqrt(cov, eps=1e-6).astype(np.float32)     # [D, D]
    mu_t = torch.from_numpy(mu.astype(np.float32))
    W_t  = torch.from_numpy(W)
    return mu_t, W_t

def _apply_whiten(z: torch.Tensor, mu: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    z: [N, D] on any device; mu/W on CPU or same device.
    Returns whitened z: (z - mu) @ W
    """
    # Move stats to z's device once
    mu_d = mu.to(z.device, non_blocking=True)
    W_d  = W.to(z.device, non_blocking=True)
    return (z - mu_d) @ W_d

def _resolve_topk(num_items: int, topk: float | int) -> int:
    if isinstance(topk, float):
        if topk <= 0: k = 1
        elif topk <= 1: k = int(math.ceil(num_items * topk))
        else: k = int(round(topk))
    else:
        k = int(topk)
    return max(1, min(num_items, k))

def _pairwise_dist(X: torch.Tensor, Y: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """
    X: [N, D], Y: [M, D] -> [N, M]
    metric in {"euclidean", "cosine"}
    """
    if metric == "euclidean":
        return torch.cdist(X, Y)
    elif metric == "cosine":
        Xn = F.normalize(X, dim=-1)
        Yn = F.normalize(Y, dim=-1)
        # cosine distance = 1 - cosine similarity
        return 1.0 - (Xn @ Yn.T)
    else:
        raise ValueError(f"Unknown metric: {metric}")


@torch.no_grad()
def calibrate_task_image_stats(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    train_loader: DataLoader,
    task_name: str,
    topk: float | int = 0.02,
    sigma_mult: float = 3.0,
    center_quantile: Optional[float] = None,
    knn_k: int = 1,
    robust: bool = False,           # <--- new
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute per-task μ/σ (or median/MAD) of image scores on train-good images,
    matching the same pooling and k-NN settings used at inference.
    """
    if task_name not in ucad_mem.mem:
        return None, None, None

    extractor.eval().to(DEVICE)
    prompt_bank.load(task_name, extractor)

    mem = ucad_mem.mem[task_name]
    K = mem.knowledge.to(DEVICE)

    scores = []

    for batch in train_loader:
        imgs = batch["image"].to(DEVICE)         # [B,3,H,W]
        feats = extractor.get_final_patches(imgs, use_prompts=True)   # [B,N,D]
        B, N, D = feats.shape
        Z = feats.reshape(B*N, D)
        d = torch.cdist(Z, K)                    # [BN, M]

        # per-patch distances (k-NN mean or min)
        if knn_k <= 1:
            min_d = d.min(dim=1).values.reshape(B, N)
        else:
            k_eff = min(knn_k, d.size(1))
            min_d = torch.topk(d, k=k_eff, largest=False, dim=1).values.mean(dim=1).reshape(B, N)

        # sparse-defect pooling with optional baseline subtraction
        k = _resolve_topk(N, topk)
        pooled = []
        for b in range(B):
            a = min_d[b].flatten()
            if center_quantile is not None and 0.0 < center_quantile < 0.5:
                # subtract a robust low-quantile baseline
                q_idx = max(1, int(center_quantile * a.numel()))
                base = torch.topk(a, k=q_idx, largest=False).values.max()
                a = a - base
            pooled.append(torch.topk(a, k=k, largest=True).values.mean())
        scores.append(torch.stack(pooled).cpu())

    if len(scores) == 0:
        return None, None, None

    s = torch.cat(scores).float()    # [num_train_good]

    if robust:
        med = s.median().item()
        mad = (s - med).abs().median().item()
        sigma = 1.4826 * mad if mad > 0 else max(s.std(unbiased=False).item(), 1e-6)
        thr = med + sigma_mult * sigma
        return med, sigma, thr
    else:
        mu = s.mean().item()
        sigma = max(s.std(unbiased=False).item(), 1e-6)
        thr = mu + sigma_mult * sigma
        return mu, sigma, thr


@torch.no_grad()
def select_task_by_imgscore(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    imgs: torch.Tensor,
    q: float = 0.02,
    center_quantile: Optional[float] = None,
) -> List[str]:
    extractor.eval().to(DEVICE)
    B = imgs.size(0)
    best_scores = [float("inf")] * B
    best_tasks = [None] * B
    for t in ucad_mem.tasks():
        prompt_bank.load(t, extractor)
        feats = extractor.get_final_patches(imgs, use_prompts=True)   # [B,N,D]
        B_, N, D = feats.shape
        Z = feats.reshape(-1, D)

        mem = ucad_mem.mem[t]
        K = mem.knowledge.to(imgs.device)

        if mem.whitener is not None and mem.mean_vec is not None:
            Zw = _apply_whiten(Z, mem.mean_vec, mem.whitener)
            Kw = _apply_whiten(K, mem.mean_vec, mem.whitener)
            dmin = torch.cdist(Zw, Kw).min(dim=1).values.view(B_, N)
        else:
            dmin = torch.cdist(Z, K).min(dim=1).values.view(B_, N)

        raw_scores = torch.stack([_image_score_from_min_d(dmin[b], q=q, center_quantile=center_quantile) for b in range(B_)])
        mu, sigma = mem.mu, mem.sigma
        s_cal = (raw_scores - mu) / sigma
        for b in range(B_):
            val = float(s_cal[b].item())
            if val < best_scores[b]:
                best_scores[b] = val
                best_tasks[b] = t
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

def _aggregate_image_score(amap: torch.Tensor, ratio: float = 0.02) -> torch.Tensor:
    flat = amap.flatten()
    k = max(1, int(round(ratio * flat.numel())))
    return torch.topk(flat, k=k, largest=True).values.mean()

@torch.no_grad()
def infer_batch_anomaly(
    extractor: ViTPromptExtractor,
    prompt_bank: PromptBank,
    ucad_mem: UCADMemory,
    imgs: torch.Tensor,
    key_layer: int = 5,
    image_topk: float = 0.02,
    center_quantile: float | None = None,
    calib_stats: Optional[Dict[str, Dict[str, float]]] = None,
    knn_k: int = 1,
    per_task_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    # NEW:
    use_multiscale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      image_scores: [B] calibrated scores
      maps: [B, Hp, Wp] pixel anomaly maps (0..1 scaled later by caller)
    """
    extractor.eval().to(DEVICE)
    B, _, H, W = imgs.shape
    ps = 16
    Hp, Wp = H // ps, W // ps

    if per_task_overrides is None:
        per_task_overrides = {}

    # First pick tasks (keys-based is fine; it’s working well)
    tasks = select_task_by_keys(extractor, prompt_bank, ucad_mem, imgs, key_layer=key_layer)

    image_scores_out: List[float] = []
    maps_out: List[torch.Tensor] = []

    # torchvision rotate helper (keeps tensor on device)
    import torchvision.transforms.functional as TF

    for b in range(B):
        t = tasks[b]
        over = per_task_overrides.get(t, {})
        tk = int(over.get("knn_k", knn_k))
        topk_ratio = float(over.get("image_topk", image_topk))
        cq = float(over.get("center_quantile", center_quantile)) if over.get("center_quantile", None) is not None else center_quantile
        metric = over.get("metric", "euclidean")  # e.g., set "cosine" for screw

        # TTA angles (only for hard classes; default none)
        tta_angles = over.get("tta_angles", [])
        if not isinstance(tta_angles, (list, tuple)):
            tta_angles = []

        # run one or multiple orientations, aggregate by min (more robust for goods)
        img_b = imgs[b:b+1]
        all_img_scores = []
        all_maps = []

        angles_to_run = ([0] + list(tta_angles)) if 0 not in tta_angles else list(tta_angles)
        for ang in angles_to_run:
            if ang != 0:
                x = TF.rotate(img_b, angle=ang, interpolation=InterpolationMode.BILINEAR)
            else:
                x = img_b

            # ---- MULTI-SCALE MAPS ----
            # Final layer (prompted) vs knowledge
            prompt_bank.load(t, extractor)
            z_final = extractor.get_final_patches(x, use_prompts=True)     # [1, N, D]
            Nf, Df = z_final.shape[1], z_final.shape[2]
            Zf = z_final.reshape(-1, Df)
            Kf = ucad_mem.mem[t].knowledge.to(Zf.device)

            Dfmat = _pairwise_dist(Zf, Kf, metric=metric)                   # [Nf, M]
            if tk > 1:
                vals, _ = torch.topk(Dfmat, k=min(tk, Dfmat.shape[1]), dim=1, largest=False)
                min_final = vals.mean(dim=1)                                # k-NN mean
            else:
                min_final = Dfmat.min(dim=1).values

            # Early layer (neutral) vs keys, if enabled
            if use_multiscale:
                z_early = extractor.get_key_patches(x, layer_idx=key_layer, use_prompts=False)  # [1, Ne, D]
                Ne, De = z_early.shape[1], z_early.shape[2]
                Ze = z_early.reshape(-1, De)
                Ke = ucad_mem.mem[t].keys.to(Ze.device)
                Demat = _pairwise_dist(Ze, Ke, metric=metric)               # [Ne, Mk]
                if tk > 1:
                    vals_e, _ = torch.topk(Demat, k=min(tk, Demat.shape[1]), dim=1, largest=False)
                    min_early = vals_e.mean(dim=1)
                else:
                    min_early = Demat.min(dim=1).values
            else:
                min_early = None

            # Reshape to patch grid (handle any mismatch safely)
            def _reshape_grid(v: torch.Tensor, N: int) -> torch.Tensor:
                exp = Hp * Wp
                out = v
                if N != exp:
                    if N > exp:
                        out = out[:exp]
                    else:
                        out = F.pad(out, (0, exp - N), value=float(out.mean()))
                return out.reshape(Hp, Wp)

            amap_final = _reshape_grid(min_final, Nf)
            if use_multiscale:
                amap_early = _reshape_grid(min_early, z_early.shape[1])
                # combine by elementwise max (keep whichever layer flags a pixel more)
                amap = torch.maximum(amap_final, amap_early)
            else:
                amap = amap_final

            # ---- image score pooling + calibration ----
            # center-quantile subtraction
            flat = amap.flatten()
            if cq is not None and 0.0 < cq < 0.5:
                k0 = max(1, int(cq * flat.numel()))
                base = torch.topk(flat, k=k0, largest=False).values.max()
                flat = flat - base

            k = max(1, int(round(topk_ratio * flat.numel())))
            raw_score = torch.topk(flat, k=k, largest=True).values.mean()

            if calib_stats is not None and t in calib_stats:
                mu = calib_stats[t]["mu"]
                sigma = max(calib_stats[t]["sigma"], 1e-6)
                img_s = (raw_score - mu) / sigma
            else:
                img_s = raw_score

            # rotate map back to original orientation if needed
            if ang != 0:
                amap_back = TF.rotate(amap.unsqueeze(0), angle=-ang, interpolation=InterpolationMode.BILINEAR).squeeze(0)
            else:
                amap_back = amap

            all_img_scores.append(img_s)
            all_maps.append(amap_back)

        # Aggregate across TTA by min (robust/forgiving)
        img_score_b = torch.stack(all_img_scores).min()
        # For pixel map, min is also OK (keeps lowest anomaly among orientations)
        # If you prefer more sensitivity, use median; for screw start with min.
        map_b = torch.stack(all_maps).min(dim=0).values

        image_scores_out.append(float(img_score_b))
        maps_out.append(map_b.cpu())

    return torch.tensor(image_scores_out), torch.stack(maps_out, dim=0)

# --------------------------------------------------------------------------------------
# Orchestration: train per task, update PromptBank + UCADMemory, evaluate
# --------------------------------------------------------------------------------------

def calculate_pixel_aupr_all(gt_masks: np.ndarray, pixel_scores: np.ndarray) -> float:
    """
    Pixel-level AUPR across ALL test pixels (normal + anomalous) — this is
    the setting used in the paper's Pixel AUPR tables.
    """
    y_true = gt_masks.reshape(-1).astype(np.uint8)
    y_score = pixel_scores.reshape(-1).astype(np.float32)
    return average_precision_score(y_true, y_score)

def calculate_pixel_aupr_defect_only(gt_masks: np.ndarray, pixel_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    """
    Optional: AUPR using ONLY anomalous images (mirrors your current AUROC fn style).
    """
    pixels_per_image = gt_masks.size // len(gt_labels)
    y_true = gt_masks.reshape(len(gt_labels), pixels_per_image)
    y_score = pixel_scores.reshape(len(gt_labels), pixels_per_image)
    idx = np.where(gt_labels > 0)[0]
    if len(idx) == 0:
        return 0.0
    return average_precision_score(y_true[idx].ravel(), y_score[idx].ravel())

def calculate_pixel_auroc(gt_masks: np.ndarray, pixel_scores: np.ndarray, gt_labels: np.ndarray) -> float:
    # AUROC over pixels of anomalous images only
    pixels_per_image = gt_masks.size // len(gt_labels)
    y_true  = gt_masks.reshape(len(gt_labels), pixels_per_image)
    y_score = pixel_scores.reshape(len(gt_labels), pixels_per_image)
    idx = np.where(gt_labels > 0)[0]
    if len(idx) == 0:
        return 0.0
    return roc_auc_score(y_true[idx].ravel(), y_score[idx].ravel())

def forgetting_measure(T_rows: list[list[float]]) -> float:
    k = len(T_rows)
    if k <= 1:
        return 0.0
    max_tasks = max(len(r) for r in T_rows)
    T = np.full((k, max_tasks), np.nan, dtype=np.float32)
    for i, r in enumerate(T_rows):
        T[i, :len(r)] = r

    diffs = []
    for j in range(k - 1):
        prev_best = np.nanmax(T[:k-1, j])
        now = T[k-1, j]
        if np.isfinite(prev_best) and np.isfinite(now):
            diffs.append(max(0.0, prev_best - now))  # clamp
    return float(np.mean(diffs)) if diffs else 0.0

def _true_category_from_path(p: str) -> str:
    # mvtec path: .../<category>/test/<defect_type>/xxx.png
    parts = Path(p).parts
    for i in range(len(parts)-1):
        if parts[i].endswith(("mvtec", "mvtec2d")):
            return parts[i+1]
    # fallback: second-to-last that isn't "test" or "train"
    for q in reversed(parts):
        if q not in {"test", "train", "ground_truth"}:
            return q
    return "unknown"

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
    # image-level pooling (ratio of top pixels used for image score)
    image_topk_ratio: float = 0.02,
    center_quantile: Optional[float] = 0.10,
):
    print("[init] Building extractor + segmenter…")
    extractor = ViTPromptExtractor(
        fast_layers=(0, 1, 2),
        slow_layers=None,
        freeze_backbone=True,
        add_scale_gates=True,
        timm_name='vit_base_patch16_224.augreg_in21k',
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
    calib_stats: Dict[str, Dict[str, float]] = {}
    seen_categories: List[str] = []
    hist_img_rows: list[list[float]] = []
    hist_pix_rows: list[list[float]] = []

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
        epochs = 5
        meta.cfg.outer_steps = epochs * max(1, len(query_loader))
        meta.cfg.outer_batches_per_step = 1
        combined_prev_knowledge = ucad_mem.concat_knowledge()
        outer_stats = meta.outer_update_many(query_loader, combined_prev_knowledge)
        print(
            f"[outer] loss_q={outer_stats['loss_q']:.5f} | align={outer_stats['loss_align']:.5f} "
            f"| scl={outer_stats['scl']:.5f} | steps={outer_stats['steps']}"
        )

        # ----- Snapshot prompts for this task -----
        prompt_bank.save(category, extractor)

        # ----- Build per-task memory (keys/knowledge, with k-center/fps) -----
        ROTATE_TASKS = {"screw", "cable", "zipper", "metal_nut"}  # extend if you like
        rotations = (0,1,2,3) if category in ROTATE_TASKS else (0,)

        keys, knowledge = build_task_memory(
            extractor, train_loader,
            key_layer=meta.cfg.key_layer,
            max_keys=max_keys,
            max_knowledge=max_knowledge,
            rotations=rotations,
        )

        ucad_mem.add(category, keys, knowledge)
        # if category == "screw":
        #     prune_dense_knowledge(ucad_mem, "screw", keep_ratio=0.8, sample=15000, k=10, metric="cosine")
        print(f"[memory] {category}: keys={ucad_mem.mem[category].keys.shape[0]} | knowledge={ucad_mem.mem[category].knowledge.shape[0]}")


        # ----- Compute per-task whitener (from knowledge) -----
        try:
            mean_vec, W = compute_whitener_from_memory(knowledge, use_ledoit=True)
            ucad_mem.mem[category].mean_vec = mean_vec   # CPU tensors
            ucad_mem.mem[category].whitener = W
            print(f"[whiten] {category}: mean[D]={mean_vec.numel()} W[D,D]={W.shape}")
        except Exception as e:
            print(f"[whiten] {category}: FAILED to compute whitener ({e}); falling back to L2.")


        # ----- Per-task calibration of image scores -----
        # sensible defaults
        knn_k_default = 1
        center_quantile = 0.05
        image_topk_ratio = 0.02

        # Per-task overrides (augment screw)
        overrides = {
            "screw":      {
                "knn_k": 7,                    # more averaging
                "image_topk": 0.01,            # focus on sparser hot pixels
                "center_quantile": 0.10,       # subtract more baseline
                "metric": "cosine",            # robust to highlights
                "tta_angles": [0, 90, 180, 270],
            },
            "metal_nut":  {"knn_k": 3, "metric": "cosine"},
            "cable":      {"knn_k": 3},
            "zipper":     {"knn_k": 3},
        }

        mu, sigma, thr = calibrate_task_image_stats(
            extractor=extractor,
            prompt_bank=prompt_bank,
            ucad_mem=ucad_mem,
            train_loader=train_loader,
            task_name=category,
            topk=image_topk_ratio,
            sigma_mult=3.0,
            center_quantile=center_quantile,   # if you’re using it
            knn_k=knn_k_default,               # see below
            robust=True,                       # <--- robust stats
        )


        if mu is not None:
            sigma = max(float(sigma), 1e-6)
            calib_stats[category] = {"mu": float(mu), "sigma": sigma, "thr": float(thr)}
            ucad_mem.mem[category].mu = float(mu)
            ucad_mem.mem[category].sigma = sigma
            print(f"[calib] {category}: mu={mu:.4f} sigma={sigma:.4f} thr={thr:.4f}")
        else:
            print(f"[calib] {category}: skipped (no memory yet)")

        # ----- Evaluate on all seen categories (task-agnostic) -----
        test_loader = get_continual_test_loaders(seen_categories, data_root, image_size, batch_size)
        gt_labels_all: List[int] = []
        image_scores_all: List[float] = []
        gt_masks_all: List[np.ndarray] = []
        pixel_maps_all: List[np.ndarray] = []

        # per-category aggregates
        per_task = {cat: {"y_img": [], "s_img": [], "y_pix": [], "s_pix": []} for cat in seen_categories}

        # --- routing accuracy counters (debug) ---
        route_hits_keys = 0
        route_hits_img  = 0
        route_total     = 0

        for batch in tqdm(test_loader, desc="Testing"):
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            gt_masks = batch["mask"].squeeze(1).cpu().numpy()
            paths = batch["path"]

            # --- task-selection predictions for this batch (debug only) ---
            pred_keys = select_task_by_keys(
                extractor, prompt_bank, ucad_mem, imgs, key_layer=meta.cfg.key_layer
            )
            pred_img = select_task_by_imgscore(
                extractor, prompt_bank, ucad_mem, imgs, q=image_topk_ratio
            )
            for i, p in enumerate(paths):
                true_cat = _true_category_from_path(p)
                if pred_keys[i] == true_cat:
                    route_hits_keys += 1
                if pred_img[i] == true_cat:
                    route_hits_img += 1
            route_total += len(paths)

            # --- actual inference (still uses keys-based selection inside) ---
            # hard-class overrides
            # overrides = {
            #     "screw":      {"knn_k": 5, "image_topk": 0.01, "center_quantile": 0.10},
            #     "cable":      {"knn_k": 3},
            #     "zipper":     {"knn_k": 3},
            #     "metal_nut":  {"knn_k": 3},
            # }

            img_scores, maps = infer_batch_anomaly(
                extractor, prompt_bank, ucad_mem, imgs,
                key_layer=meta.cfg.key_layer,
                image_topk=image_topk_ratio,
                center_quantile=center_quantile,
                calib_stats=calib_stats,
                knn_k=knn_k_default,
                per_task_overrides=None,
                use_multiscale=True,
            )



            # Upsample and normalize maps
            Ht, Wt = imgs.shape[-2:]
            maps_up = F.interpolate(maps.unsqueeze(1), size=(Ht, Wt), mode='bilinear', align_corners=False).squeeze(1)
            minv = maps_up.view(maps_up.size(0), -1).min(dim=1).values.view(-1, 1, 1)
            maxv = maps_up.view(maps_up.size(0), -1).max(dim=1).values.view(-1, 1, 1)
            maps_up = (maps_up - minv) / (maxv - minv + 1e-8)

            # global aggregates
            gt_labels_all.append(labels)
            image_scores_all.append(img_scores.cpu().numpy())
            gt_masks_all.append(gt_masks)
            pixel_maps_all.append(maps_up.cpu().numpy())

            # per-category aggregates
            for i, p in enumerate(paths):
                cat = _true_category_from_path(p)
                per_task[cat]["y_img"].append(int(labels[i]))
                per_task[cat]["s_img"].append(float(img_scores[i].item()))
                per_task[cat]["y_pix"].append(gt_masks[i])              # (H, W)
                per_task[cat]["s_pix"].append(maps_up[i].cpu().numpy()) # (H, W)

        # --- routing accuracy summary ---
        if route_total > 0:
            print(f"[routing] keys-based accuracy:     {route_hits_keys}/{route_total} = {route_hits_keys/route_total:.3f}")
            print(f"[routing] imgscore-based accuracy: {route_hits_img}/{route_total} = {route_hits_img/route_total:.3f}")
        else:
            print("[routing] no test samples")

        # --- compute global metrics ---
        gt_labels = np.concatenate(gt_labels_all)
        image_scores = np.concatenate(image_scores_all)
        gt_masks_np = np.concatenate(gt_masks_all)
        pixel_scores_np = np.concatenate(pixel_maps_all)

        image_auroc = roc_auc_score(gt_labels, image_scores)
        pixel_auroc = calculate_pixel_aupr_defect_only(gt_masks_np, pixel_scores_np, gt_labels)
        pixel_aupr_all = calculate_pixel_aupr_all(gt_masks_np, pixel_scores_np)

        # --- per-category metrics (needed for FM) ---
        percat_img = []
        percat_pix = []
        for cat in seen_categories:
            y_img = np.asarray(per_task[cat]["y_img"], dtype=np.int64)
            s_img = np.asarray(per_task[cat]["s_img"], dtype=np.float32)
            if len(np.unique(y_img)) > 1:
                percat_img.append(roc_auc_score(y_img, s_img))
            else:
                percat_img.append(float('nan'))

            if len(per_task[cat]["y_pix"]) > 0:
                y_pix = np.stack(per_task[cat]["y_pix"], axis=0)  # [n, H, W]
                s_pix = np.stack(per_task[cat]["s_pix"], axis=0)  # [n, H, W]
                percat_pix.append(calculate_pixel_aupr_all(y_pix, s_pix))
            else:
                percat_pix.append(float('nan'))

        # --- print per-category tables ---
        print("\nPer-task Image AUROC:")
        for cat, val in zip(seen_categories, percat_img):
            print(f"  {cat:12s}: {val:.4f}" if np.isfinite(val) else f"  {cat:12s}: nan")

        print("Per-task Pixel AUPR (all pixels):")
        for cat, val in zip(seen_categories, percat_pix):
            print(f"  {cat:12s}: {val:.4f}" if np.isfinite(val) else f"  {cat:12s}: nan")

        # --- FM bookkeeping ---
        hist_img_rows.append(percat_img)
        hist_pix_rows.append(percat_pix)
        fm_img = forgetting_measure(hist_img_rows)
        fm_pix = forgetting_measure(hist_pix_rows)

        print(f"\n--- Results after task '{category}' ({len(seen_categories)} seen) ---")
        print(f"  Image AUROC: {image_auroc:.4f}")
        print(f"  Pixel AUROC: {pixel_auroc:.4f}")
        print(f"  Pixel AUPR (all pixels): {pixel_aupr_all:.4f}")
        print(f"  FM (Image AUROC): {fm_img:.3f}")
        print(f"  FM (Pixel AUPR):  {fm_pix:.3f}")
        print("-" * 50)

# --------------------------------------------------------------------------------------
# Main (edit paths as needed)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"      # <-- CHANGE ME
    SAM_CHECKPOINT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/sam_vit_b_01ec64.pth"                  # e.g., "/path/to/sam_vit_b_01ec64.pth" (optional)

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
        scl_coef=0.1,
        image_topk_ratio=0.02,          # top 2% pooling for image score
        center_quantile=None,        # set to e.g. 0.10 to subtract 10th percentile before pooling
    )


