# Load Dataset (Capsule)
# pip install torch torchvision pillow
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"

class MVTecAD(Dataset):
    """
    root/
      └── capsule/
           ├── train/good/*.png
           ├── test/{good,crack,...}/*.png
           └── ground_truth/{crack,...}/*_mask.png
    """
    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",             # "train" or "test"
        image_size: int = 256,
        transform: Optional[T.Compose] = None,
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size

        base = self.root / category
        assert base.exists(), f"Category not found: {base}"

        # --------- collect image & mask paths ----------
        img_paths: List[Path] = []
        mask_paths: List[Optional[Path]] = []
        labels: List[int] = []           # 0 = good, 1 = defective
        defect_types: List[str] = []

        if split == "train":
            for p in sorted((base / "train" / "good").glob("*.*")):
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    img_paths.append(p)
                    mask_paths.append(None)   # no masks for train
                    labels.append(0)
                    defect_types.append("good")
        elif split == "test":
            gt_root = base / "ground_truth"
            for sub in sorted((base / "test").iterdir()):
                if not sub.is_dir():
                    continue
                dt = sub.name  # "good" or a defect type
                for p in sorted(sub.glob("*.*")):
                    if p.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                        continue
                    img_paths.append(p)
                    defect_types.append(dt)
                    if dt == "good":
                        mask_paths.append(None)
                        labels.append(0)
                    else:
                        # ground_truth/<defect>/{stem}_mask.png (allow any suffix)
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

        # --------- transforms ----------
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size, InterpolationMode.BILINEAR),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.transform = transform

        # separate mask pipeline (NEAREST to keep binary edges)
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
            # zeros when there's no mask (train/good and test/good)
            H, W = target_size
            return torch.zeros((1, H, W), dtype=torch.float32)

        m = Image.open(mp).convert("L")  # 8-bit mask
        m = self.mask_resize(m)
        m = self.mask_center(m)
        m = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(m.tobytes()))
                              .view(m.height, m.width).numpy() > 0).astype("float32"))
        # reshape to (1,H,W)
        return m.unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img = self._load_image(self.img_paths[idx])
        img = self.transform(img)  # (3,H,W)
        _, H, W = img.shape

        mask = self._load_mask(idx, (H, W))
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        return {
            "image": img,                      # (3,H,W), normalized
            "mask": mask,                      # (1,H,W), 0/1
            "label": label,                    # 0 good, 1 defect
            "path": str(self.img_paths[idx]),
            "defect_type": self.defect_types[idx],
        }


def load_mvtec_category(
    root: str,
    category: str = "capsule",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """Create train/test datasets and dataloaders for one MVTec category."""
    train_ds = MVTecAD(root, category, split="train", image_size=image_size)
    test_ds  = MVTecAD(root, category, split="test",  image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    return train_ds, test_ds, train_loader, test_loader

train_ds, test_ds, train_ld, test_ld = load_mvtec_category(
    DATA_ROOT, "capsule",
    image_size=256,
    batch_size=8,
    num_workers=0,          # <<< key change
    pin_memory=False,       # optional on CPU/MPS
)
batch = next(iter(train_ld))

import torch.nn as nn
import timm
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Add this to the top of your script with the other imports
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu") 
print(f"Using device: {DEVICE}")

# ======================================================================================

import torch
import torch.nn as nn
from torchvision.models import wide_resnet50_2

import torch.nn as nn
from torchvision.models import wide_resnet50_2
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name, layers_to_extract_from, common_size=(16, 16)):
        super().__init__()
        backbone = wide_resnet50_2(weights='IMAGENET1K_V1')
        
        self.initial_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        
        self.feature_extractor = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.common_size = common_size

        layer_map = {
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4,
        }
        
        # Output channel dimensions for each layer
        output_dims = {'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        
        # Desired common channel dimension
        common_dim = 256 

        # Add layers to be extracted and their projectors
        for layer_name in layers_to_extract_from:
            self.feature_extractor[layer_name] = layer_map[layer_name]
            self.projectors[layer_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d(common_size), # Resize to common spatial size
                nn.Conv2d(output_dims[layer_name], common_dim, kernel_size=1) # Project to common channel dim
            )

    def forward(self, x):
        x = self.initial_layers(x)
        processed_features = []
        for name, layer in self.feature_extractor.items():
            x = layer(x)
            # Project features to a common dimension and spatial size
            projected_x = self.projectors[name](x)
            processed_features.append(projected_x)

        # Reshape for concatenation
        # The tensors in processed_features are (B, common_dim, H, W)
        # We want to concatenate them into a single tensor of shape (B, N_PATCHES, C)
        combined_features = torch.cat(processed_features, dim=1)
        
        # Reshape to (B, N_PATCHES, C) format
        B, C_combined, H, W = combined_features.shape
        N_PATCHES = H * W
        
        # Reshape and permute to match the PatchCore algorithm's expected format
        return combined_features.permute(0, 2, 3, 1).view(B, N_PATCHES, C_combined)

# The correct implementation for `train_patchcore`
# Modified train_patchcore function
def train_patchcore(train_loader, feature_extractor):
    memory_bank = []
    
    # Switch to evaluation mode
    feature_extractor.eval()
    
    with torch.no_grad():
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            
            # Extract features. This returns a LIST of tensors
            list_of_patch_features = feature_extractor(images) 
            
            # Now, iterate through the list and process each tensor
            for patch_features in list_of_patch_features:
                # Reshape each tensor and append to memory bank
                memory_bank.append(patch_features.view(-1, patch_features.shape[-1]))
            
    # Concatenate all features into a single tensor
    memory_bank = torch.cat(memory_bank, dim=0)
    
    return memory_bank

def test_patchcore(
    test_loader: DataLoader,
    feature_extractor: FeatureExtractor,
    memory_bank: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tests the PatchCore model.
    
    Returns:
        Tuple: (Ground truth labels, Predicted image scores, Ground truth masks, Predicted pixel scores)
    """
    print("Testing...")
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
            masks = batch["mask"].squeeze(1).numpy() # (B, H, W)
            
            # Extract features
            patch_features = feature_extractor(images) # (B, N_PATCHES, C)
            B, N_PATCHES, C = patch_features.shape
            
            # Reshape for distance calculation
            # Use .reshape() instead of .view() to handle non-contiguous tensors
            patch_features = patch_features.reshape(-1, C) # (B * N_PATCHES, C)

            # Calculate nearest neighbor distances
            # torch.cdist is efficient for this
            distances = torch.cdist(patch_features, memory_bank) # (B * N_PATCHES, M_coreset)
            min_distances, _ = torch.min(distances, dim=1) # (B * N_PATCHES)
            
            # Reshape back to image-like format
            anomaly_map = min_distances.view(B, int(N_PATCHES**0.5), int(N_PATCHES**0.5))
            
            # Image-level anomaly score is the max of the patch scores
            img_scores_batch = torch.max(anomaly_map.view(B, -1), dim=1)[0]
            
            # Upsample anomaly map to match original image size for pixel-level evaluation
            H_orig, W_orig = images.shape[-2:]
            anomaly_map_resized = F.interpolate(
                anomaly_map.unsqueeze(1),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze(1) # (B, H, W)

            # Store results
            gt_labels.append(labels.numpy())
            image_scores.append(img_scores_batch.cpu().numpy())
            gt_masks.append(masks)
            pixel_scores.append(anomaly_map_resized.cpu().numpy())

    return (
        np.concatenate(gt_labels),
        np.concatenate(image_scores),
        np.concatenate(gt_masks),
        np.concatenate(pixel_scores),
    )

if __name__ == '__main__':
    # --- Configuration ---
    DATASET_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d" # IMPORTANT: Change this to your dataset path
    CATEGORY = "zipper"  
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    
    # --- Load Data ---
    train_ds, test_ds, train_loader, test_loader = load_mvtec_category(
        root=DATASET_ROOT,
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,          # <<< key change
        pin_memory=False,       # optional on CPU/MPS
    )

    # --- Initialize Model ---
    # You can experiment with different layers
    feature_extractor = FeatureExtractor(
        backbone_name='wide_resnet50_2',
        layers_to_extract_from=['layer2', 'layer3']
    )
    
    # --- Run Training ---
    # This builds the memory bank
    memory_bank = train_patchcore(train_loader, feature_extractor)

    # --- Run Testing ---
    # This compares test images to the memory bank
    gt_labels, image_scores, gt_masks, pixel_scores = test_patchcore(
        test_loader, feature_extractor, memory_bank
    )
    
    # --- Evaluate ---
    # Flatten masks and scores for pixel-level AUROC
    gt_masks_flat = gt_masks.flatten()
    pixel_scores_flat = pixel_scores.flatten()

    # Filter out 'good' images from pixel-level evaluation
    # as they don't have masks
    gt_masks_flat = gt_masks_flat[gt_labels.repeat(IMAGE_SIZE*IMAGE_SIZE) == 1]
    pixel_scores_flat = pixel_scores_flat[gt_labels.repeat(IMAGE_SIZE*IMAGE_SIZE) == 1]
    
    # Calculate AUROC
    image_auroc = roc_auc_score(gt_labels, image_scores)
    pixel_auroc = roc_auc_score(gt_masks_flat, pixel_scores_flat)

    print(f"\n--- Results for category: {CATEGORY} ---")
    print(f"Image-level AUROC: {image_auroc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auroc:.4f}")