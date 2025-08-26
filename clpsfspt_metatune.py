# Load Dataset (Capsule)
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import wide_resnet50_2
from torch.utils.data._utils.collate import default_collate
import math
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import cv2
import random

# Add these for SAM
from segment_anything import sam_model_registry, SamPredictor

# --- Your existing data and model setup ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- MVTecAD Dataset Class (remains the same) ---
class MVTecAD(Dataset):
    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 256,
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
                if not sub.is_dir(): continue
                dt = sub.name
                for p in sorted(sub.glob("*.*")):
                    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]: continue
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
        img = Image.open(path).convert("RGB")
        return img

    def _load_mask(self, idx: int, target_size: Tuple[int, int]) -> torch.Tensor:
        mp = self.mask_paths[idx]
        if mp is None:
            H, W = target_size
            return torch.zeros((1, H, W), dtype=torch.float32)
        m = Image.open(mp).convert("L")
        m = self.mask_resize(m)
        m = self.mask_center(m)
        m_np = np.array(m, dtype=np.uint8)
        m_tensor = torch.from_numpy((m_np > 0).astype("float32"))
        return m_tensor.unsqueeze(0)


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

# --- Helper functions (remain the same) ---
def load_mvtec_category(
    root: str,
    category: str = "capsule",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    train_ds = MVTecAD(root, category, split="train", image_size=image_size)
    test_ds = MVTecAD(root, category, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=train_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=test_collate_fn
    )
    return train_ds, test_ds, train_loader, test_loader

def _normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m, M = x.min(), x.max()
    return (x - m) / (M - m + eps)

def _topk_peaks(heat: np.ndarray, k: int = 3, nms_kernel: int = 9) -> List[Tuple[int, int]]:
    H, W = heat.shape
    peaks = []
    heat_copy = heat.copy()
    for _ in range(k):
        idx = np.argmax(heat_copy)
        y, x = divmod(idx, W)
        peaks.append((y, x))
        y0 = max(0, y - nms_kernel // 2); y1 = min(H, y + nms_kernel // 2 + 1)
        x0 = max(0, x - nms_kernel // 2); x1 = min(W, x + nms_kernel // 2 + 1)
        heat_copy[y0:y1, x0:x1] = -np.inf
    return peaks

def _sample_negatives(heat: np.ndarray, n: int = 4, low_q: float = 0.2) -> List[Tuple[int, int]]:
    H, W = heat.shape
    thr = np.quantile(heat, low_q)
    candidates = np.argwhere(heat <= thr)
    pts = []
    if len(candidates) > 0:
        vals = heat[candidates[:,0], candidates[:,1]]
        order = np.argsort(vals)
        for i in order[:n]:
            y, x = map(int, candidates[i])
            pts.append((y, x))
    pts += [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]
    seen = set(); uniq = []
    for y,x in pts:
        if (y,x) not in seen:
            uniq.append((y,x)); seen.add((y,x))
    return uniq[:max(n,4)]

def _box_from_heat(heat: np.ndarray, hi_q: float = 0.97) -> Optional[Tuple[int,int,int,int]]:
    H, W = heat.shape
    thr = np.quantile(heat, hi_q)
    mask = heat >= thr
    if mask.sum() == 0:
        return None
    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return None
    r = max(regions, key=lambda rr: rr.area)
    minr, minc, maxr, maxc = r.bbox
    return (int(minc), int(minr), int(maxc), int(maxr))

def _alignment_score(binary_mask: np.ndarray, heat: np.ndarray, thresh: float = 0.7) -> float:
    heat_n = _normalize01(heat)
    if binary_mask.dtype != np.bool_:
        bm = binary_mask > 0.5
    else:
        bm = binary_mask
    if bm.sum() == 0:
        return -1e9
    inside = heat_n[bm].mean()
    outside = heat_n[~bm].mean()
    heat_bin = heat_n >= thresh
    inter = float((bm & heat_bin).sum())
    union = float((bm | heat_bin).sum() + 1e-8)
    iou = inter / union
    return inside - outside + 0.2 * iou

def _scale_points(points: List[Tuple[int,int]], src_hw: Tuple[int,int], dst_hw: Tuple[int,int]) -> np.ndarray:
    srcH, srcW = src_hw; dstH, dstW = dst_hw
    out = []
    for (y, x) in points:
        yy = y * (dstH / srcH)
        xx = x * (dstW / srcW)
        out.append([xx, yy])
    return np.array(out, dtype=np.float32)

def _scale_box(box_xyxy: Tuple[int,int,int,int], src_hw: Tuple[int,int], dst_hw: Tuple[int,int]) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy
    srcH, srcW = src_hw; dstH, dstW = dst_hw
    X0 = x0 * (dstW / srcW); Y0 = y0 * (dstH / srcH)
    X1 = x1 * (dstW / srcW); Y1 = y1 * (dstH / srcH)
    return np.array([X0, Y0, X1, Y1], dtype=np.float32)

def _choose_best_mask(masks: np.ndarray, heat: np.ndarray) -> np.ndarray:
    best, best_s = None, -1e9
    H_heat, W_heat = heat.shape
    for m in masks:
        if m.dtype == bool:
            m = m.astype(np.uint8) * 255
        resized_m = cv2.resize(m, (W_heat, H_heat), interpolation=cv2.INTER_NEAREST)
        s = _alignment_score(resized_m, heat)
        if s > best_s:
            best, best_s = resized_m, s
    if best is None:
        return np.zeros_like(heat, dtype=np.float32)
    return best.astype(np.float32)

def _system1_prompts(heat: np.ndarray, k_pos: int = 3, n_neg: int = 4, smooth_sigma: float = 1.5):
    Hs = gaussian_filter(heat, sigma=smooth_sigma)
    pos_pts = _topk_peaks(Hs, k=k_pos)
    neg_pts = _sample_negatives(Hs, n=n_neg)
    box = _box_from_heat(Hs, hi_q=0.97)
    return pos_pts, neg_pts, box

def _system2_refine(predictor: SamPredictor,
                    image_np: np.ndarray,
                    heat: np.ndarray,
                    pos_pts_xy: np.ndarray,
                    neg_pts_xy: np.ndarray,
                    init_mask: np.ndarray,
                    max_iter: int = 2) -> np.ndarray:
    Hh, Wh = heat.shape
    curr_mask = init_mask.copy()
    pos = pos_pts_xy.copy()
    neg = neg_pts_xy.copy()
    predictor.set_image(image_np)

    for _ in range(max_iter):
        curr_mask_n = _normalize01(curr_mask)
        resid = heat - curr_mask_n
        y_pos, x_pos = np.unravel_index(np.argmax(resid), resid.shape)
        masked_low = curr_mask > 0.5
        if masked_low.any():
            inv = np.where(masked_low, -heat, 1.0)
            y_neg, x_neg = np.unravel_index(np.argmax(inv), inv.shape)
        else:
            y_neg, x_neg = 0, 0

        Himg, Wimg = image_np.shape[:2]
        new_pos = _scale_points([(y_pos, x_pos)], (Hh, Wh), (Himg, Wimg))
        new_neg = _scale_points([(y_neg, x_neg)], (Hh, Wh), (Himg, Wimg))

        pos = np.vstack([pos, new_pos])
        neg = np.vstack([neg, new_neg])

        masks, _, _ = predictor.predict(
            point_coords=np.vstack([pos, neg]),
            point_labels=np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int64),
            multimask_output=True
        )
        cand = _choose_best_mask(masks, heat)
        if _alignment_score(cand, heat) >= _alignment_score(curr_mask, heat) + 1e-5:
            curr_mask = cand
        else:
            break
    return curr_mask

# --- OPTIMIZED: FeatureExtractor with Fast/Slow Prompts ---
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, layers_to_extract_from,
                 common_size=(16, 16),
                 fast_prompt_size=8, slow_prompt_size=8,
                 common_dim=256):
        super().__init__()
        backbone = wide_resnet50_2(weights='IMAGENET1K_V1')
        self.initial_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.feature_extractor = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.common_size = common_size
        layer_map = {'layer2': backbone.layer2, 'layer3': backbone.layer3, 'layer4': backbone.layer4}
        output_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        self.common_dim = common_dim

        for layer_name in layers_to_extract_from:
            self.feature_extractor[layer_name] = layer_map[layer_name]
            self.projectors[layer_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(common_size),
                nn.Conv2d(output_dims[layer_name], self.common_dim, kernel_size=1)
            )

        # Brain-inspired fast and slow prompt embeddings
        self.fast_prompt_embeddings = nn.Parameter(torch.randn(1, fast_prompt_size, self.common_dim))
        self.slow_prompt_embeddings = nn.Parameter(torch.randn(1, slow_prompt_size, self.common_dim))
        self.num_prompts = fast_prompt_size + slow_prompt_size

    def forward(self, x):
        x = self.initial_layers(x)
        processed_features = []
        for name, layer in self.feature_extractor.items():
            x = layer(x)
            projected_x = self.projectors[name](x)
            processed_features.append(projected_x)

        stacked_features = torch.stack(processed_features, dim=0)
        combined_features = torch.sum(stacked_features, dim=0)

        B, C, H, W = combined_features.shape
        N_PATCHES = H * W
        patch_features = combined_features.permute(0, 2, 3, 1).reshape(B, N_PATCHES, C)

        # Concatenate the fast and slow prompts to the patch features
        fast_prompt = self.fast_prompt_embeddings.expand(B, -1, -1)
        slow_prompt = self.slow_prompt_embeddings.expand(B, -1, -1)
        combined_output = torch.cat([slow_prompt, fast_prompt, patch_features], dim=1)
        return combined_output

def train_patchcore(train_loader, feature_extractor):
    """Builds the memory bank using the current state of the feature extractor."""
    memory_bank = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Building Memory Bank"):
            images = batch["image"].to(DEVICE)
            # We only use patch features for the memory bank, not prompts
            patch_features = feature_extractor(images)[:, feature_extractor.num_prompts:, :]
            for features in patch_features:
                memory_bank.append(features.reshape(-1, features.shape[-1]))
    memory_bank = torch.cat(memory_bank, dim=0)
    return memory_bank


# --- NEW: Meta-Learning Training Function for Prompts ---
def train_with_meta_learning(
    feature_extractor: FeatureExtractor,
    train_loader: DataLoader,
    memory_bank: Optional[torch.Tensor],
    epochs: int = 5,
    inner_lr: float = 0.03,
    outer_lr: float = 1e-4,
    inner_steps: int = 3
):
    """
    Updates the fast and slow prompts using a meta-learning strategy.
    """
    print("Starting meta-learning for fast and slow prompts...")
    feature_extractor.train() # Set model to training mode

    # Freeze backbone, only prompts are trainable
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.fast_prompt_embeddings.requires_grad = True
    feature_extractor.slow_prompt_embeddings.requires_grad = True

    # Optimizers for fast (inner loop) and slow (outer loop) prompts
    optimizer_slow = torch.optim.Adam([feature_extractor.slow_prompt_embeddings], lr=outer_lr)

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Meta-Training Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            
            batch_size = images.shape[0]
            if batch_size < 2: continue
            split_idx = batch_size // 2
            support_images = images[:split_idx]
            query_images = images[split_idx:]
            
            # --- INNER LOOP: Update Fast Prompts ---
            temp_fast_prompts = feature_extractor.fast_prompt_embeddings.clone().detach().requires_grad_(True)
            optimizer_fast = torch.optim.SGD([temp_fast_prompts], lr=inner_lr)

            for _ in range(inner_steps):
                optimizer_fast.zero_grad()
                
                B, C_img, H_img, W_img = support_images.shape
                with torch.no_grad():
                    initial_out = feature_extractor.initial_layers(support_images)
                    processed_feats = []
                    x_temp = initial_out
                    # *** FIXED BUG HERE ***
                    # Iterate over items (name, layer) not values
                    for name, layer in feature_extractor.feature_extractor.items():
                        x_temp = layer(x_temp)
                        processed_feats.append(feature_extractor.projectors[name](x_temp))
                    
                    combined_features = torch.sum(torch.stack(processed_feats, dim=0), dim=0)
                    C_feat = combined_features.shape[1]
                    patch_features = combined_features.permute(0, 2, 3, 1).reshape(B, -1, C_feat)

                fast_prompt = temp_fast_prompts.expand(B, -1, -1)
                slow_prompt = feature_extractor.slow_prompt_embeddings.expand(B, -1, -1)
                
                full_features = torch.cat([slow_prompt, fast_prompt, patch_features], dim=1)
                
                dists = torch.cdist(full_features.view(-1, C_feat), memory_bank)
                min_d, _ = torch.min(dists, dim=1)
                
                anomaly_scores = torch.amax(min_d.reshape(B, -1), dim=1)
                loss_fast = torch.mean(anomaly_scores)
                
                loss_fast.backward()
                optimizer_fast.step()

            # --- OUTER LOOP: Update Slow Prompts ---
            optimizer_slow.zero_grad()
            feature_extractor.fast_prompt_embeddings.data = temp_fast_prompts.data

            B, C_feat, H, W = query_images.shape
            full_features_query = feature_extractor(query_images)
            
            dists_query = torch.cdist(full_features_query.reshape(-1, C_feat), memory_bank)
            min_d_query, _ = torch.min(dists_query, dim=1)
            anomaly_scores_query = torch.amax(min_d_query.reshape(B, -1), dim=1)
            loss_slow = torch.mean(anomaly_scores_query)
            
            loss_slow.backward()
            optimizer_slow.step()

            total_loss += loss_slow.item()
            pbar.set_postfix({"loss": f"{loss_slow.item():.4f}"})
        
        print(f"Epoch {epoch+1} average meta-loss: {total_loss / len(train_loader):.4f}")

    feature_extractor.eval()
    return feature_extractor


def test_patchcore_sam(
    test_loader: DataLoader,
    feature_extractor: FeatureExtractor,
    memory_bank: torch.Tensor,
    sam_predictor: SamPredictor,
    k_pos: int = 3,
    n_neg: int = 4,
    smooth_sigma: float = 1.5,
    max_refine_iter: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Testing with PatchCore + SAM (Fast&Slow prompts)...")
    feature_extractor.eval()
    feature_extractor.to(DEVICE)
    memory_bank = memory_bank.to(DEVICE)

    gt_labels = []
    image_scores = []
    gt_masks = []
    pixel_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"].cpu().numpy()
            gt_masks_batch = batch["mask"].squeeze(1).numpy()
            original_pil_images = batch["image_pil"]

            patch_features_full = feature_extractor(images)
            
            B, _, C = patch_features_full.shape
            
            dists = torch.cdist(patch_features_full.reshape(-1, C), memory_bank)
            min_d, _ = torch.min(dists, dim=1)
            
            side_len = int(math.sqrt(patch_features_full.shape[1] - feature_extractor.num_prompts))
            patch_scores = min_d.reshape(B, -1)[:, feature_extractor.num_prompts:]
            anomaly_map = patch_scores.reshape(B, side_len, side_len)

            Ht, Wt = images.shape[-2:]
            anomaly_map_resized = F.interpolate(
                anomaly_map.unsqueeze(1), size=(Ht, Wt), mode='bilinear', align_corners=False
            ).squeeze(1)
            image_scores_batch = torch.amax(anomaly_map_resized.view(B, -1), dim=1).cpu().numpy()
            final_masks = []
            for i in range(B):
                heat = anomaly_map_resized[i].cpu().numpy()
                if labels[i] == 0:
                    final_masks.append(np.zeros((Ht, Wt), dtype=np.float32))
                    continue

                pos_pts, neg_pts, box_h = _system1_prompts(heat, k_pos=k_pos, n_neg=n_neg, smooth_sigma=smooth_sigma)
                image_np = np.array(original_pil_images[i])
                Himg, Wimg = image_np.shape[:2]
                pos_xy_img = _scale_points(pos_pts, (Ht, Wt), (Himg, Wimg))
                neg_xy_img = _scale_points(neg_pts, (Ht, Wt), (Himg, Wimg))
                box_xyxy_img = _scale_box(box_h, (Ht, Wt), (Himg, Wimg)) if box_h is not None else None
                sam_predictor.set_image(image_np)
                pt_coords = np.vstack([pos_xy_img, neg_xy_img])
                pt_labels = np.concatenate([np.ones(len(pos_xy_img)), np.zeros(len(neg_xy_img))]).astype(np.int64)
                masks_list = []
                for bx in [None, box_xyxy_img]:
                    try:
                        masks, _, _ = sam_predictor.predict(
                            point_coords=pt_coords,
                            point_labels=pt_labels,
                            box=bx,
                            multimask_output=True
                        )
                        masks_list.append(masks.astype(np.float32))
                    except Exception:
                        pass
                if len(masks_list) == 0:
                    fast_mask = np.zeros((Ht, Wt), dtype=np.float32)
                else:
                    candidates = [cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST) for m in np.vstack(masks_list)]
                    fast_mask = _choose_best_mask(np.stack(candidates), heat)
                refined = _system2_refine(
                    predictor=sam_predictor,
                    image_np=image_np,
                    heat=heat,
                    pos_pts_xy=pos_xy_img,
                    neg_pts_xy=neg_xy_img,
                    init_mask=fast_mask,
                    max_iter=max_refine_iter
                )
                final_masks.append(refined.astype(np.float32))
            gt_labels.append(labels)
            image_scores.append(image_scores_batch)
            gt_masks.append(gt_masks_batch)
            fm = np.stack(final_masks)
            Hn = np.stack([_normalize01(h) for h in anomaly_map_resized.cpu().numpy()])
            pixel_scores.append(Hn * fm)

    return (
        np.concatenate(gt_labels),
        np.concatenate(image_scores),
        np.concatenate(gt_masks),
        np.concatenate(pixel_scores),
    )


def train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch = {
        "image": default_collate([d["image"] for d in batch]),
        "path": [d["path"] for d in batch],
    }
    return collated_batch

def test_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pil_images = [d["image_pil"] for d in batch]
    for item in batch:
        del item["image_pil"]
    collated_tensors = default_collate(batch)
    collated_tensors["image_pil"] = pil_images
    return collated_tensors

# --- New: Continual learning functions ---
def get_continual_test_loaders(seen_categories: List[str], data_root: str, image_size: int, batch_size: int):
    """
    Creates a combined test loader for all categories seen so far.
    """
    test_datasets = [MVTecAD(data_root, category, split="test", image_size=image_size) for category in seen_categories]
    combined_test_ds = torch.utils.data.ConcatDataset(test_datasets)
    return DataLoader(
        combined_test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=test_collate_fn
    )

def coreset_subsampling(memory_bank: torch.Tensor, max_size: int) -> torch.Tensor:
    """
    Performs coreset subsampling to reduce the memory bank size.
    This is a simplified version using random sampling.
    """
    if memory_bank.shape[0] <= max_size:
        return memory_bank
    
    indices = random.sample(range(memory_bank.shape[0]), max_size)
    return memory_bank[indices]

def run_continual_learning(
    categories: List[str],
    data_root: str,
    image_size: int,
    batch_size: int,
    sam_checkpoint: str,
    max_memory_bank_size: int = 100000,
):
    """
    Orchestrates the entire continual learning pipeline with meta-learning prompts.
    """
    sam_predictor = SamPredictor(sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device=DEVICE))
    
    feature_extractor = FeatureExtractor(
        backbone_name='wide_resnet50_2',
        layers_to_extract_from=['layer2', 'layer3']
    ).to(DEVICE)
    
    combined_memory_bank = None
    seen_categories = []

    for task_idx, category in enumerate(categories):
        print(f"\n--- Starting Task {task_idx+1}/{len(categories)}: {category} ---")
        seen_categories.append(category)

        # Step 1: Load data for the new category
        _, _, train_loader, _ = load_mvtec_category(
            root=data_root, category=category, image_size=image_size, batch_size=batch_size
        )
        
        # Step 2: Adapt feature extractor using meta-learning if not the first task
        if combined_memory_bank is not None and len(combined_memory_bank) > 0:
            feature_extractor = train_with_meta_learning(
                feature_extractor, train_loader, combined_memory_bank.to(DEVICE)
            )

        # Step 3: Build/update the memory bank with features from the adapted model
        current_task_memory = train_patchcore(train_loader, feature_extractor)

        if combined_memory_bank is None:
            combined_memory_bank = current_task_memory
        else:
            combined_memory_bank = torch.cat([combined_memory_bank.cpu(), current_task_memory.cpu()], dim=0)
        
        # Coreset subsampling to manage memory bank size
        combined_memory_bank = coreset_subsampling(combined_memory_bank, max_memory_bank_size)
        print(f"Memory bank size after task {task_idx+1}: {combined_memory_bank.shape[0]}")

        # Step 4: Evaluate on all seen categories
        test_loader = get_continual_test_loaders(seen_categories, data_root, image_size, batch_size)
        
        gt_labels, image_scores, gt_masks, pixel_scores = test_patchcore_sam(
            test_loader, feature_extractor, combined_memory_bank, sam_predictor
        )
        
        # --- Corrected Performance Evaluation ---
        # Image-level AUROC
        image_auroc = roc_auc_score(gt_labels, image_scores)

        # Pixel-level AUROC (only on anomalous images)
        # Find indices of images that are not "good"
        anomaly_indices = np.where(gt_labels == 1)[0]
        if len(anomaly_indices) > 0:
            gt_masks_anomalous = gt_masks[anomaly_indices].flatten()
            pixel_scores_anomalous = pixel_scores[anomaly_indices].flatten()
            pixel_auroc = roc_auc_score(gt_masks_anomalous, pixel_scores_anomalous)
        else:
            pixel_auroc = np.nan # No anomalous samples to evaluate on

        print(f"\n--- Cumulative Results after training on {category} ---")
        print(f"Categories evaluated: {seen_categories}")
        print(f"Image-level AUROC: {image_auroc:.4f}")
        print(f"Pixel-level AUROC: {pixel_auroc:.4f}")


if __name__ == '__main__':
    # --- Configuration for continual learning ---
    CATEGORIES_SEQUENCE = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper"
    ]
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    
    sam_checkpoint = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/sam_vit_b_01ec64.pth"

    run_continual_learning(
        categories=CATEGORIES_SEQUENCE,
        data_root=DATA_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        sam_checkpoint=sam_checkpoint
    ) 