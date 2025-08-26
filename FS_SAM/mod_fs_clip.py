#!/usr/bin/env python3
import os
import random
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ucad_dataset import UCADDataset


def sample_episode(dataset, class_to_indices, n_way=3, n_shot=1, n_query=1):
    
    sup_imgs, sup_lbls, qry_imgs, qry_lbls = [], [], [], []
    ways = random.sample(list(class_to_indices.keys()), n_way)
    for i, c in enumerate(ways):
        indices = class_to_indices[c]
        # ensure enough examples exist:
        assert len(indices) >= n_shot + n_query, (
            f"class {c} only has {len(indices)} samples, "
            f"needs {n_shot+n_query}"
        )
        picked = random.sample(indices, n_shot + n_query)
        # support
        for p in picked[:n_shot]:
            img, _ = dataset[p]
            sup_imgs.append(img)
            sup_lbls.append(i)
        # query
        for p in picked[n_shot:]:
            img, _ = dataset[p]
            qry_imgs.append(img)
            qry_lbls.append(i)

    # stack into batches
    sup_batch = torch.stack(sup_imgs)   # [n_way*n_shot, C, H, W]
    qry_batch = torch.stack(qry_imgs)   # [n_way*n_query, C, H, W]
    sup_lbls = torch.tensor(sup_lbls)
    qry_lbls = torch.tensor(qry_lbls)
    return sup_batch, sup_lbls, qry_batch, qry_lbls

if __name__ == "__main__":

    # for reproducibility
    import random, numpy as np, torch
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    freeze_support()

    # --- Configuration ---
    device = torch.device("cpu")
    data_root = "/Users/lenguyenlinhdan/Downloads/FSCIL_TCDS/UCAD-main/mvtec2d-sam-b"
    subdatasets = ["bottle","cable","capsule","carpet","grid","hazelnut"]

    # OpenCLIP transforms for ViT-B/32
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # --- 0) Datasets & Index Maps ---
    train_ds = UCADDataset(data_root, subdatasets, split="train", transform=transform)
    test_ds  = UCADDataset(data_root, subdatasets, split="test",  transform=transform)

    # Build class→list-of-indices mappings separately
    train_class_to_indices = {i: [] for i in range(len(subdatasets))}
    for idx, (_path, lbl) in enumerate(train_ds.samples):
        train_class_to_indices[lbl].append(idx)

    test_class_to_indices = {i: [] for i in range(len(subdatasets))}
    for idx, (_path, lbl) in enumerate(test_ds.samples):
        test_class_to_indices[lbl].append(idx)

    # for batch eval later
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

    # --- 1) Load & Freeze OpenCLIP ---
    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai', device=device
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_dim = model.text_projection.shape[1]  # e.g. 512

    # --- 2) Prompt Vectors & Optimizers ---
    n_slow, n_fast = 4, 2
    slow_prompts = nn.Parameter(torch.randn(n_slow, embed_dim) * 0.02)
    fast_prompts = nn.Parameter(torch.randn(n_fast, embed_dim) * 0.02)

    inner_opt = optim.SGD([fast_prompts], lr=1e-2)
    outer_opt = optim.SGD([slow_prompts], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def encode_image(x):
        return model.encode_image(x)

    def encode_prompt(c):
        txt = subdatasets[c]
        tokens = tokenizer([txt]).to(device)
        base_emb = model.encode_text(tokens)  # [1, dim]
        combo = base_emb \
              + slow_prompts.mean(0, keepdim=True) \
              + fast_prompts.mean(0, keepdim=True)
        return combo.squeeze(0)  # → [dim]

    # --- 3) Meta-Training (3-way 1-shot) ---
    n_epochs = 50
    for epoch in range(1, n_epochs+1):
        sup_imgs, sup_lbls, qry_imgs, qry_lbls = sample_episode(
            train_ds, train_class_to_indices,
            n_way=3, n_shot=1, n_query=1
        )

        # Inner (fast) updates
        fast_prompts.data.zero_()
        for _ in range(5):
            inner_opt.zero_grad()
            img_feats = encode_image(sup_imgs)
            txt_feats = torch.stack([encode_prompt(int(c)) for c in sup_lbls])
            logits = img_feats @ txt_feats.t()
            loss = loss_fn(logits, sup_lbls)
            loss.backward()
            inner_opt.step()

        # Outer (slow) updates
        fast_prompts.requires_grad_(False)
        outer_opt.zero_grad()
        img_feats_q = encode_image(qry_imgs)
        txt_feats_q = torch.stack([encode_prompt(int(c)) for c in sup_lbls])
        loss_q = loss_fn(img_feats_q @ txt_feats_q.t(), qry_lbls)
        loss_q.backward()
        outer_opt.step()
        fast_prompts.requires_grad_(True)

        print(f"Epoch {epoch}/{n_epochs} — meta-loss: {loss_q.item():.4f}")

    # --- 4) Test-time adaptation & evaluation ---
    sup_imgs, sup_lbls, qry_imgs, qry_lbls = sample_episode(
        test_ds, test_class_to_indices,
        n_way=3, n_shot=1, n_query=1
    )

    fast_prompts.data.zero_()
    for _ in range(5):
        inner_opt.zero_grad()
        img_feats = encode_image(sup_imgs)
        txt_feats = torch.stack([encode_prompt(int(c)) for c in sup_lbls])
        logits = img_feats @ txt_feats.t()
        loss = loss_fn(logits, sup_lbls)
        loss.backward()
        inner_opt.step()

    preds = (encode_image(qry_imgs) @ torch.stack([encode_prompt(int(c)) for c in sup_lbls]).t()).argmax(dim=1)
    acc = (preds == qry_lbls).float().mean().item()
    print(f"Few-shot accuracy on test episode: {acc:.4f}")