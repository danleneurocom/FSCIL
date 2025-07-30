#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# SAM registry
from segment_anything import sam_model_registry

class MVTecBinary(Dataset):
    """Image‐level binary dataset for one MVTec-2D category."""
    def __init__(self, root_dir, category, split="train", transform=None):
        """
        Expects:
          root_dir/
            <category>/
              train/good/*.png
              test/{good,crack,poke,…}/*.png
              ground_truth/…      <-- ignored
        """
        self.samples = []
        self.transform = transform
        base = os.path.join(root_dir, category, split)
        for sub in os.listdir(base):
            subdir = os.path.join(base, sub)
            if not os.path.isdir(subdir):
                continue
            label = 0 if sub == "good" else 1
            for fname in os.listdir(subdir):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(subdir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",       type=str, required=True,
                        help="Path to mvtec2d-sam-b")
    parser.add_argument("--category",        type=str, default="capsule")
    parser.add_argument("--sam_type",        type=str, default="vit_b")
    parser.add_argument("--sam_checkpoint",  type=str, required=True)
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--lr",              type=float, default=1e-3)
    args = parser.parse_args()

    # reproducibility
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) transforms
    transform = transforms.Compose([
        # SAM expects 1024×1024 so its pos_embed lines up
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # 2) dataset & loaders
    train_ds = MVTecBinary(args.data_root, args.category, split="train", transform=transform)
    test_ds  = MVTecBinary(args.data_root, args.category, split="test",  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Train / Test sizes: {len(train_ds)}/{len(test_ds)}")

    # 3) load SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    for p in sam.parameters(): p.requires_grad = False

    # feature dim = sam.image_encoder.neck[0].out_channels
    embed_dim = sam.image_encoder.neck[0].out_channels
    print("SAM pooled feature dim:", embed_dim)

    # 4) binary head
    head = nn.Linear(embed_dim, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(head.parameters(), lr=args.lr)

    # helper to extract pooled features
    @torch.no_grad()
    def extract_feats(x):
        fmap = sam.image_encoder(x)          # [B, C, 64, 64]
        return fmap.mean(dim=(2,3))         # [B, C]

    # 5) training loop
    for epoch in range(1, args.epochs+1):
        head.train()
        total_loss = 0.0
        for imgs, labs in train_loader:
            imgs = imgs.to(device)
            labs = labs.float().to(device)

            feats = extract_feats(imgs)
            logits = head(feats).squeeze(1)
            loss = criterion(logits, labs)
            opt.zero_grad(); loss.backward(); opt.step()

            total_loss += loss.item() * len(imgs)
        print(f"Epoch {epoch}/{args.epochs} — train loss: {total_loss/len(train_ds):.3f}")

    # 6) evaluation
    head.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(device)
            feats = extract_feats(imgs)
            logits = head(feats).squeeze(1).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labs.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # metrics
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs > 0.5).astype(int)
    acc   = accuracy_score(all_labels, preds)
    roc   = roc_auc_score(all_labels, probs)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Test ROC-AUC:    {roc:.3f}")

if __name__ == "__main__":
    main()
