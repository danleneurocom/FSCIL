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
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate
import math
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

# Add these for SAM
from segment_anything import sam_model_registry, SamPredictor
import cv2

# --- Your existing data and model setup ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"
CATEGORY = "capsule"
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- MVTecAD Dataset Class ---
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
        m = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(m.tobytes()))
                              .view(m.height, m.width).numpy() > 0).astype("float32"))
        return m.unsqueeze(0)

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
            "image_pil": img_pil, # Add the PIL image here
        }

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
        collate_fn=train_collate_fn # <<< Use train collate here
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=test_collate_fn # <<< Use test collate here
    )
    return train_ds, test_ds, train_loader, test_loader

def _normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m, M = x.min(), x.max()
    return (x - m) / (M - m + eps)

def _topk_peaks(heat: np.ndarray, k: int = 3, nms_kernel: int = 9) -> List[Tuple[int, int]]:
    """Return top-k (y,x) peaks after simple NMS on heat (H×W)."""
    H, W = heat.shape
    peaks = []
    heat_copy = heat.copy()
    for _ in range(k):
        idx = np.argmax(heat_copy)
        y, x = divmod(idx, W)
        peaks.append((y, x))
        # zero-out a local window to avoid duplicates
        y0 = max(0, y - nms_kernel // 2); y1 = min(H, y + nms_kernel // 2 + 1)
        x0 = max(0, x - nms_kernel // 2); x1 = min(W, x + nms_kernel // 2 + 1)
        heat_copy[y0:y1, x0:x1] = -np.inf
    return peaks

def _sample_negatives(heat: np.ndarray, n: int = 4, low_q: float = 0.2) -> List[Tuple[int, int]]:
    """Pick negatives from lowest-quantile heat (and edges to discourage leakage)."""
    H, W = heat.shape
    thr = np.quantile(heat, low_q)
    candidates = np.argwhere(heat <= thr)
    pts = []
    if len(candidates) > 0:
        # random but reproducible ordering by value then y then x
        vals = heat[candidates[:,0], candidates[:,1]]
        order = np.argsort(vals)
        for i in order[:n]:
            y, x = map(int, candidates[i])
            pts.append((y, x))
    # add 4 corners as strong negatives (helps SAM not expand to borders)
    pts += [(0,0), (0,W-1), (H-1,0), (H-1,W-1)]
    # dedup and cap
    seen = set(); uniq = []
    for y,x in pts:
        if (y,x) not in seen:
            uniq.append((y,x)); seen.add((y,x))
    return uniq[:max(n,4)]

def _box_from_heat(heat: np.ndarray, hi_q: float = 0.97) -> Optional[Tuple[int,int,int,int]]:
    """Return (x0, y0, x1, y1) of the tight box around the high-heat blob; None if empty."""
    H, W = heat.shape
    thr = np.quantile(heat, hi_q)
    mask = heat >= thr
    if mask.sum() == 0:
        return None
    labeled = label(mask)
    # pick the largest region
    regions = regionprops(labeled)
    if not regions:
        return None
    r = max(regions, key=lambda rr: rr.area)
    minr, minc, maxr, maxc = r.bbox
    # SAM wants (x0, y0, x1, y1)
    return (int(minc), int(minr), int(maxc), int(maxr))

def _alignment_score(binary_mask: np.ndarray, heat: np.ndarray, thresh: float = 0.7) -> float:
    """Higher is better: cover high heat, avoid low heat."""
    heat_n = _normalize01(heat)
    if binary_mask.dtype != np.bool_:
        bm = binary_mask > 0.5
    else:
        bm = binary_mask
    if bm.sum() == 0:
        return -1e9
    inside = heat_n[bm].mean()
    outside = heat_n[~bm].mean()
    # optional IoU with hard heat threshold to discourage bloated masks
    heat_bin = heat_n >= thresh
    inter = float((bm & heat_bin).sum())
    union = float((bm | heat_bin).sum() + 1e-8)
    iou = inter / union
    return inside - outside + 0.2 * iou

def _scale_points(points: List[Tuple[int,int]], src_hw: Tuple[int,int], dst_hw: Tuple[int,int]) -> np.ndarray:
    """Scale (y,x) points from src (H,W) grid to dst (H,W) image coords (SAM expects x,y)."""
    srcH, srcW = src_hw; dstH, dstW = dst_hw
    out = []
    for (y, x) in points:
        yy = y * (dstH / srcH)
        xx = x * (dstW / srcW)
        out.append([xx, yy])  # (x,y)
    return np.array(out, dtype=np.float32)

def _scale_box(box_xyxy: Tuple[int,int,int,int], src_hw: Tuple[int,int], dst_hw: Tuple[int,int]) -> np.ndarray:
    """Scale (x0,y0,x1,y1) from src grid to dst image coords."""
    x0, y0, x1, y1 = box_xyxy
    srcH, srcW = src_hw; dstH, dstW = dst_hw
    X0 = x0 * (dstW / srcW); Y0 = y0 * (dstH / srcH)
    X1 = x1 * (dstW / srcW); Y1 = y1 * (dstH / srcH)
    return np.array([X0, Y0, X1, Y1], dtype=np.float32)

def _choose_best_mask(masks: np.ndarray, heat: np.ndarray) -> np.ndarray:
    """masks: (N,H,W) bool/float; choose by alignment score."""
    best, best_s = None, -1e9
    
    # Get the target dimensions from the heat map
    H_heat, W_heat = heat.shape

    for m in masks:
        # Check if the mask is boolean and convert to uint8
        if m.dtype == bool:
            m = m.astype(np.uint8) * 255
            
        # Resize the mask to match the heat map dimensions
        resized_m = cv2.resize(m, (W_heat, H_heat), interpolation=cv2.INTER_NEAREST)
        s = _alignment_score(resized_m, heat)
        if s > best_s:
            best, best_s = resized_m, s
    
    # Handle the case where no valid mask was found
    if best is None:
        return np.zeros_like(heat, dtype=np.float32)

    return best.astype(np.float32)

def _system1_prompts(heat: np.ndarray, k_pos: int = 3, n_neg: int = 4, smooth_sigma: float = 1.5):
    """Fast prompts (pos/neg points + optional box) on heat grid."""
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
    """Iteratively add informative points based on residuals."""
    Hh, Wh = heat.shape
    curr_mask = init_mask.copy()
    pos = pos_pts_xy.copy()
    neg = neg_pts_xy.copy()
    predictor.set_image(image_np)

    for _ in range(max_iter):
        # residual where heat suggests more mask
        curr_mask_n = _normalize01(curr_mask)
        resid = heat - curr_mask_n
        # next positive where resid is largest
        y_pos, x_pos = np.unravel_index(np.argmax(resid), resid.shape)
        # next negative where mask covers low heat
        masked_low = curr_mask > 0.5
        if masked_low.any():
            inv = np.where(masked_low, -heat, 1.0)  # high where we'd like negatives
            y_neg, x_neg = np.unravel_index(np.argmax(inv), inv.shape)
        else:
            y_neg, x_neg = 0, 0

        # scale to image coords
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
        # accept if improved
        if _alignment_score(cand, heat) >= _alignment_score(curr_mask, heat) + 1e-5:
            curr_mask = cand
        else:
            break
    return curr_mask

# --- FeatureExtractor and PatchCore training/testing functions ---
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, layers_to_extract_from, common_size=(16, 16)):
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
        common_dim = 256
        for layer_name in layers_to_extract_from:
            self.feature_extractor[layer_name] = layer_map[layer_name]
            self.projectors[layer_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(common_size),
                nn.Conv2d(output_dims[layer_name], common_dim, kernel_size=1)
            )

    def forward(self, x):
        x = self.initial_layers(x)
        processed_features = []
        for name, layer in self.feature_extractor.items():
            x = layer(x)
            projected_x = self.projectors[name](x)
            processed_features.append(projected_x)
        combined_features = torch.cat(processed_features, dim=1)
        B, C_combined, H, W = combined_features.shape
        N_PATCHES = H * W
        return combined_features.permute(0, 2, 3, 1).view(B, N_PATCHES, C_combined)

def train_patchcore(train_loader, feature_extractor):
    memory_bank = []
    feature_extractor.eval()
    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            # ... (rest of the function is the same)
            patch_features = feature_extractor(images)
            for features in patch_features:
                memory_bank.append(features.view(-1, features.shape[-1]))
    memory_bank = torch.cat(memory_bank, dim=0)
    return memory_bank

# ======================================================================================
# Corrected test_patchcore_sam Function
# ======================================================================================
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

    gt_labels = []
    image_scores = []
    gt_masks = []
    pixel_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)               # (B,3,Ht,Wt) transformed to 256x256
            labels = batch["label"].cpu().numpy()            # (B,)
            gt_masks_batch = batch["mask"].squeeze(1).numpy()# (B,Ht,Wt)
            original_pil_images = batch["image_pil"]         # list of PILs (orig HW)

            # --- Stage 1: PatchCore image/pixel anomaly ---
            patch_features = feature_extractor(images)       # (B,Np,C)
            B, N_PATCHES, C = patch_features.shape
            feats_flat = patch_features.reshape(-1, C)       # (B*Np, C)
            # cdist may be heavy; chunk if needed
            dists = torch.cdist(feats_flat, memory_bank)     # (B*Np, |M|)
            min_d, _ = torch.min(dists, dim=1)
            side = int(math.sqrt(N_PATCHES))
            anomaly_map = min_d.view(B, side, side)          # (B,Sh,Sw)

            # upsample to tensor size (e.g., 256x256) for scoring
            Ht, Wt = images.shape[-2:]
            anomaly_map_resized = F.interpolate(
                anomaly_map.unsqueeze(1), size=(Ht, Wt), mode='bilinear', align_corners=False
            ).squeeze(1)                                     # (B,Ht,Wt)

            # image-level score: max heat
            image_scores_batch = torch.amax(anomaly_map_resized.view(B, -1), dim=1).cpu().numpy()

            # --- Stage 2: SAM prompting (Fast & Slow) ---
            final_masks = []
            for i in range(B):
                heat = anomaly_map_resized[i].cpu().numpy()  # (Ht,Wt) in transformed grid
                if labels[i] == 0:
                    final_masks.append(np.zeros((Ht, Wt), dtype=np.float32))
                    continue

                # FAST: seeds (pos/neg points + optional box) on heat grid (Ht,Wt)
                pos_pts, neg_pts, box_h = _system1_prompts(heat, k_pos=k_pos, n_neg=n_neg, smooth_sigma=smooth_sigma)

                # scale prompts to original image coords for SAM
                image_np = np.array(original_pil_images[i])
                Himg, Wimg = image_np.shape[:2]
                pos_xy_img = _scale_points(pos_pts, (Ht, Wt), (Himg, Wimg))
                neg_xy_img = _scale_points(neg_pts, (Ht, Wt), (Himg, Wimg))
                box_xyxy_img = _scale_box(box_h, (Ht, Wt), (Himg, Wimg)) if box_h is not None else None

                sam_predictor.set_image(image_np)
                # Build prompt
                pt_coords = np.vstack([pos_xy_img, neg_xy_img])
                pt_labels = np.concatenate([np.ones(len(pos_xy_img)), np.zeros(len(neg_xy_img))]).astype(np.int64)

                masks_list = []
                # try with and without box (ensembling 1-2 calls)
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
                    # pick best among all candidates by alignment to HEAT (rescaled back to (Ht,Wt))
                    candidates = [cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST) for m in np.vstack(masks_list)]
                    fast_mask = _choose_best_mask(np.stack(candidates), heat)

                # SLOW: refine with 2–3 active iterations
                refined = _system2_refine(
                    predictor=sam_predictor,
                    image_np=image_np,
                    heat=heat,
                    pos_pts_xy=pos_xy_img,
                    neg_pts_xy=neg_xy_img,
                    init_mask=fast_mask,
                    max_iter=max_refine_iter
                )
                # refined is (Ht,Wt) float (since we compare on heat grid)
                final_masks.append(refined.astype(np.float32))

            # Store results
            gt_labels.append(labels)
            image_scores.append(image_scores_batch)
            gt_masks.append(gt_masks_batch)

            # pixel scores = heat * refined_mask
            fm = np.stack(final_masks)                        # (B,Ht,Wt)
            Hn = np.stack([_normalize01(h) for h in anomaly_map_resized.cpu().numpy()])
            pixel_scores.append(Hn * fm)

    return (
        np.concatenate(gt_labels),
        np.concatenate(image_scores),
        np.concatenate(gt_masks),
        np.concatenate(pixel_scores),
    )


def pad_mask_to_match_target(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pads a mask with zeros to match a target shape."""
    h_target, w_target = target_shape
    h_current, w_current = mask.shape
    
    if h_current == h_target and w_current == w_target:
        return mask
    
    padded_mask = np.zeros(target_shape, dtype=mask.dtype)
    padded_mask[:h_current, :w_current] = mask
    return padded_mask

def train_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A custom collate function for PatchCore training that only keeps necessary tensors.
    """
    # Create a new batch with only the data needed for training
    collated_batch = {
        "image": default_collate([d["image"] for d in batch]),
        "path": [d["path"] for d in batch],
    }
    return collated_batch

def test_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A custom collate function for PatchCore + SAM testing.
    """
    # Separate PIL images from the batch and handle other tensors normally
    pil_images = [d["image_pil"] for d in batch]
    # Remove PIL images from the dictionaries so default_collate doesn't crash
    for item in batch:
        del item["image_pil"]
    
    collated_tensors = default_collate(batch)
    collated_tensors["image_pil"] = pil_images # Add the list of PIL images back
    
    return collated_tensors

if __name__ == '__main__':
    # --- Configuration ---
    DATASET_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"
    CATEGORY = "bottle"
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    
    # --- Initialize SAM Model ---
    sam_checkpoint = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # --- Load Data ---
    train_ds, test_ds, train_loader, test_loader = load_mvtec_category(
        root=DATA_ROOT,
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    # --- Initialize PatchCore Model ---
    feature_extractor = FeatureExtractor(
        backbone_name='wide_resnet50_2',
        layers_to_extract_from=['layer2', 'layer3']
    )
    
    # --- Run Training ---
    memory_bank = train_patchcore(train_loader, feature_extractor)

    # --- Run Testing with PatchCore + SAM ---
    gt_labels, image_scores, gt_masks, pixel_scores = test_patchcore_sam(
        test_loader, feature_extractor, memory_bank, sam_predictor
    )
    
    # --- Evaluate ---
    gt_masks_flat = gt_masks.flatten()
    pixel_scores_flat = pixel_scores.flatten()

    gt_masks_flat = gt_masks_flat[gt_labels.repeat(IMAGE_SIZE*IMAGE_SIZE) == 1]
    pixel_scores_flat = pixel_scores_flat[gt_labels.repeat(IMAGE_SIZE*IMAGE_SIZE) == 1]
    
    image_auroc = roc_auc_score(gt_labels, image_scores)
    pixel_auroc = roc_auc_score(gt_masks_flat, pixel_scores_flat)

    print(f"\n--- Results for category: {CATEGORY} ---")
    print(f"Image-level AUROC: {image_auroc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")