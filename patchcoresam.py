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
    sam_predictor: SamPredictor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("Testing with PatchCore + SAM...")
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    gt_labels = []
    image_scores = []
    gt_masks = []
    pixel_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch["image"].to(DEVICE)
            labels = batch["label"]
            gt_masks_batch = batch["mask"].squeeze(1).numpy()
            original_pil_images = batch["image_pil"]
            
            # --- Stage 1: PatchCore Anomaly Detection ---
            patch_features = feature_extractor(images)
            B, N_PATCHES, C = patch_features.shape
            
            patch_features_flat = patch_features.reshape(-1, C)
            distances = torch.cdist(patch_features_flat, memory_bank)
            min_distances, _ = torch.min(distances, dim=1)
            
            anomaly_map = min_distances.view(B, int(N_PATCHES**0.5), int(N_PATCHES**0.5))
            
            H_orig, W_orig = images.shape[-2:]
            anomaly_map_resized = F.interpolate(
                anomaly_map.unsqueeze(1),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            image_scores_batch = torch.max(anomaly_map_resized.view(B, -1), dim=1)[0]
            
            # --- Stage 2: Prompting SAM ---
            sam_masks_batch = []
            
            # Find the maximum possible shape in the batch to pad to
            max_h, max_w = 0, 0
            for i in range(B):
                img_np = np.array(original_pil_images[i])
                h, w, _ = img_np.shape
                if h > max_h: max_h = h
                if w > max_w: max_w = w
            
            for i in range(B):
                single_anomaly_map = anomaly_map_resized[i].cpu().numpy()
                
                if labels[i] == 0:
                    sam_mask = np.zeros((H_orig, W_orig), dtype=np.float32)
                else:
                    max_score_idx = np.argmax(single_anomaly_map)
                    max_score_y = max_score_idx // W_orig
                    max_score_x = max_score_idx % W_orig
                    
                    image_np = np.array(original_pil_images[i])
                    sam_predictor.set_image(image_np)
                    
                    input_point = np.array([[max_score_x, max_score_y]])
                    input_label = np.array([1])
                    
                    masks, _, _ = sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    
                    sam_mask = masks[0].astype(np.float32)
                    sam_mask = cv2.resize(sam_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

                sam_masks_batch.append(sam_mask)
            
            # Pad all masks in the batch to a consistent shape before stacking
            padded_sam_masks = [pad_mask_to_match_target(m, (H_orig, W_orig)) for m in sam_masks_batch]

            # Store results
            gt_labels.append(labels.numpy())
            image_scores.append(image_scores_batch.cpu().numpy())
            gt_masks.append(gt_masks_batch)
            pixel_scores.append(np.stack(padded_sam_masks))

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
    CATEGORY = "capsule"
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