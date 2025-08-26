# mvtec_loader.py
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

class MVTecAD(Dataset):
    def __init__(self, root, category, split="train", image_size=256, transform=None):
        base = Path(root) / category
        self.split, self.image_size = split, image_size
        self.transform = transform or T.Compose([
            T.Resize(image_size, InterpolationMode.BILINEAR),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize(image_size, InterpolationMode.NEAREST),
            T.CenterCrop(image_size),
        ])

        self.imgs, self.masks, self.labels, self.types = [], [], [], []
        if split == "train":
            for p in sorted((base/"train"/"good").glob("*.*")):
                if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
                    self.imgs.append(p); self.masks.append(None)
                    self.labels.append(0); self.types.append("good")
        elif split == "test":
            gt = base/"ground_truth"
            for sub in sorted((base/"test").iterdir()):
                if not sub.is_dir(): continue
                dtyp = sub.name
                for p in sorted(sub.glob("*.*")):
                    if p.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}: continue
                    self.imgs.append(p); self.types.append(dtyp)
                    if dtyp == "good":
                        self.masks.append(None); self.labels.append(0)
                    else:
                        stem = p.stem
                        cand = sorted((gt/dtyp).glob(f"{stem}*"))
                        self.masks.append(cand[0] if cand else None)
                        self.labels.append(1)
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        x = self.transform(img)  # (3,H,W)
        _, H, W = x.shape

        mp = self.masks[i]
        if mp is None:
            m = torch.zeros(1, H, W, dtype=torch.float32)
        else:
            m_img = Image.open(mp).convert("L")
            m_img = self.mask_tf(m_img)
            m = torch.from_numpy((np.array(m_img) > 0).astype("float32")).unsqueeze(0)

        y = torch.tensor(self.labels[i], dtype=torch.long)
        return {"image": x, "mask": m, "label": y, "path": str(self.imgs[i]), "defect_type": self.types[i]}

def load_mvtec_category(root, category="capsule", image_size=256, batch_size=8, num_workers=4, pin_memory=True):
    train_ds = MVTecAD(root, category, "train", image_size)
    test_ds  = MVTecAD(root, category, "test",  image_size)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers>0))
    test_ld  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=(num_workers>0))
    return train_ds, test_ds, train_ld, test_ld
