# Load Dataset (Capsule)
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image
import torch
import clip
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from typing import List, Dict, Any, Tuple
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate


# --- Your existing data and model setup ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DATA_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d"
CATEGORY = "capsule"
DEVICE = torch.device("cpu") 
print(f"Using device: {DEVICE}")

# Load the pre-trained CLIP model
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# --- Set up text prompts ---
good_prompt = f"a photo of a good {CATEGORY}"
defective_prompt = f"a photo of a defective {CATEGORY}"
prompts = [good_prompt, defective_prompt]
text_tokens = clip.tokenize(prompts).to(DEVICE)

# Get the text features from the CLIP model
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# --- Updated MVTecAD Dataset Class ---
class MVTecAD(Dataset):
    """
    Returns both the pre-processed tensor and the original PIL image.
    """
    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 256,
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

        # Transforms for your custom tensor output
        self.transform = T.Compose([
            T.Resize(image_size, InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

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
        # Load the original image as a PIL Image
        img_pil = self._load_image(self.img_paths[idx])
        
        # Apply your custom transform for a tensor output
        img_tensor = self.transform(img_pil)
        _, H, W = img_tensor.shape
        
        mask = self._load_mask(idx, (H, W))
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        return {
            "image_tensor": img_tensor,
            "image_pil": img_pil,  # Return the PIL image
            "mask": mask,
            "label": label,
            "path": str(self.img_paths[idx]),
            "defect_type": self.defect_types[idx],
        }

# --- load_mvtec_category and test_clip_baseline are unchanged ---
def load_mvtec_category(
    root: str,
    category: str = "capsule",
    image_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    train_ds = MVTecAD(root, category, split="train", image_size=image_size)
    test_ds  = MVTecAD(root, category, split="test",  image_size=image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        collate_fn=custom_collate_fn # <<< Pass the custom function here
    )
    return train_ds, test_ds, train_loader, test_loader

def test_clip_baseline(test_loader, model, preprocess, text_features):
    print("Running CLIP baseline...")
    model.eval()
    
    gt_labels = []
    image_scores = []
    
    with torch.no_grad():
        # The batch now contains a list of PIL images and a dict of tensors
        for pil_images, batch_tensors in tqdm(test_loader, desc="Testing with CLIP"):
            labels = batch_tensors["label"]
            
            # Preprocess the list of PIL images using CLIP's preprocessor
            clip_images = torch.stack([preprocess(img) for img in pil_images]).to(DEVICE)
            
            image_features = model.encode_image(clip_images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            good_sim = similarity[:, 0]
            defective_sim = similarity[:, 1]
            score = defective_sim / (defective_sim + good_sim)
            
            gt_labels.append(labels.numpy())
            image_scores.append(score.cpu().numpy())
    
    return np.concatenate(gt_labels), np.concatenate(image_scores)

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    A custom collate function that separates PIL images from other data.
    """
    # Separate PIL images into their own list
    pil_images = [item["image_pil"] for item in batch]
    
    # Remove the PIL images from the batch dictionary so default_collate can handle the rest
    for item in batch:
        del item["image_pil"]
    
    # Use the default collate function on the remaining tensor data
    collated_tensors = default_collate(batch)
    
    return pil_images, collated_tensors

if __name__ == '__main__':
    # --- Configuration ---
    DATASET_ROOT = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/mvtec2d" 
    CATEGORY = "capsule"
    IMAGE_SIZE = 256
    BATCH_SIZE = 8

    # --- Load Data ---
    train_ds, test_ds, train_loader, test_loader = load_mvtec_category(
        root=DATASET_ROOT,
        category=CATEGORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    # The problematic line has been removed from here.

    # --- Run CLIP Baseline ---
    gt_labels, image_scores = test_clip_baseline(
        test_loader, model, preprocess, text_features
    )
    
    # --- Evaluate ---
    image_auroc = roc_auc_score(gt_labels, image_scores)
    print(f"\n--- Results for category: {CATEGORY} (CLIP Baseline) ---")
    print(f"Image-level AUROC: {image_auroc:.4f}")