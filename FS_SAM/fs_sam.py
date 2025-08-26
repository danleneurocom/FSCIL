#!/usr/bin/env python3
import os
import random
import numpy as np
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from segment_anything import sam_model_registry

class UCADDataset(Dataset):
    """UCAD loader for mvtec2d-sam-b/<class>/(train|test)/<subclass>/*.png."""
    def __init__(self, root_dir, subdatasets, split="train", transform=None):
        self.samples = []
        for idx, cat in enumerate(subdatasets):
            split_dir = os.path.join(root_dir, cat, split)
            if not os.path.isdir(split_dir): continue
            for subclass in os.listdir(split_dir):
                sd = os.path.join(split_dir, subclass)
                if not os.path.isdir(sd): continue
                for fname in os.listdir(sd):
                    if fname.lower().endswith(".png"):
                        self.samples.append((os.path.join(sd, fname), idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

def sample_episode(dataset, class_to_idx, n_way=3, n_shot=3, n_query=1):
    sup, sup_lbls, qry, qry_lbls = [], [], [], []
    ways = random.sample(list(class_to_idx), n_way)
    for i, c in enumerate(ways):
        inds = class_to_idx[c]
        picked = random.sample(inds, n_shot + n_query)
        for p in picked[:n_shot]:
            img,_ = dataset[p]; sup.append(img); sup_lbls.append(i)
        for p in picked[n_shot:]:
            img,_ = dataset[p]; qry.append(img); qry_lbls.append(i)
    sup = torch.stack(sup); qry = torch.stack(qry)
    return sup, torch.tensor(sup_lbls), qry, torch.tensor(qry_lbls)

if __name__ == "__main__":
    freeze_support()
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    device = torch.device("cpu")
    data_root = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/mvtec2d-sam-b"
    subdatasets = ["bottle","cable","capsule","carpet","grid","hazelnut"]

    # ★ HERE’S THE CRITICAL CHANGE ★
    # Resize to 1024×1024 so SAM’s pos_embed (which is learned at 1024/16=64 grid)
    # lines up exactly with a 64×64 token map.
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # Datasets + index maps
    train_ds = UCADDataset(data_root, subdatasets, split="train", transform=transform)
    test_ds  = UCADDataset(data_root, subdatasets, split="test",  transform=transform)
    train_map = {i: [] for i in range(len(subdatasets))}
    test_map  = {i: [] for i in range(len(subdatasets))}
    for idx, (_p, l) in enumerate(train_ds.samples): train_map[l].append(idx)
    for idx, (_p, l) in enumerate(test_ds.samples):  test_map[l].append(idx)

    # Load & freeze SAM
    sam_type       = "vit_b"
    sam_checkpoint = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/sam_vit_b_01ec64.pth"
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint).to(device)
    sam.eval()
    # get SAM’s feature‐map channel count from neck[0].out_channels
    embed_dim = sam.image_encoder.neck[0].out_channels

    # Prompt vectors + class prototypes
    n_slow, n_fast = 4, 2
    slow_prompts  = nn.Parameter(torch.randn(n_slow,  embed_dim)*0.02)
    fast_prompts  = nn.Parameter(torch.randn(n_fast,  embed_dim)*0.02)
    class_prompts = nn.Parameter(torch.randn(len(subdatasets), embed_dim)*0.02)

    inner_opt = optim.SGD([fast_prompts],               lr=1e-2)
    outer_opt = optim.SGD([slow_prompts, class_prompts], lr=1e-3)
    loss_fn    = nn.CrossEntropyLoss()

    def encode_image(x):
        with torch.no_grad():
            feat_map = sam.image_encoder(x)            # [B,C,64,64]
            return feat_map.mean(dim=[2,3])           # [B,C]

    def encode_prompt(c):
        cls = class_prompts[c].unsqueeze(0)         # [1,C]
        return (cls 
                + slow_prompts.mean(0,keepdim=True)
                + fast_prompts.mean(0,keepdim=True)
               ).squeeze(0)                         # [C]

    # Meta‐training loop
    for epoch in range(1, 51):
        sup, s_lbls, qry, q_lbls = sample_episode(train_ds, train_map)
        fast_prompts.data.zero_()
        for _ in range(5):
            inner_opt.zero_grad()
            feats = encode_image(sup)                   # [3,C]
            allc  = torch.stack([encode_prompt(i) for i in range(len(subdatasets))])  # [K,C]
            logits= feats @ allc.t()                    # [3,K]
            loss  = loss_fn(logits, s_lbls)
            loss.backward()
            inner_opt.step()
        # outer
        fast_prompts.requires_grad_(False)
        outer_opt.zero_grad()
        feats_q = encode_image(qry)
        loss_q  = loss_fn(feats_q @ allc.t(), q_lbls)
        loss_q.backward()
        outer_opt.step()
        fast_prompts.requires_grad_(True)
        print(f"Epoch {epoch:02d} — meta-loss {loss_q.item():.4f}")

    # Test‐time adaptation
    sup, s_lbls, qry, q_lbls = sample_episode(test_ds, test_map)
    fast_prompts.data.zero_()
    for _ in range(5):
        inner_opt.zero_grad()
        feats = encode_image(sup)
        logits= feats @ allc.t()
        loss  = loss_fn(logits, s_lbls)
        loss.backward()
        inner_opt.step()
    preds = (encode_image(qry) @ allc.t()).argmax(dim=1)
    acc   = (preds == q_lbls).float().mean().item()
    print(f"Test episode accuracy: {acc:.4f}")
