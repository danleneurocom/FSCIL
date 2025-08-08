#!/usr/bin/env python3
"""
test_capsule.py

Anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
– If train/defect/ exists, uses those defects.
– Otherwise grabs defects from test/* for training.
Evaluates few-shot on test/good vs. test/defects.
Reports both Accuracy and AUROC.
"""

import os
import random
import argparse
import numpy as np
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# allow truncated PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

from segment_anything import sam_model_registry

# ─── Training Dataset ───────────────────────────────────────────────────
class TrainDataset(Dataset):
    """
    Loads:
      - always: train/good/*.png → label 0
      - if train/defect exists: train/defect/*.png → label 1
      - else: test/<cls>/*.png for cls!="good" → label 1
    """
    def __init__(self, data_root, transform=None):
        self.transform = transform
        self.samples = []

        # 1) all good
        good_dir = os.path.join(data_root, "train", "good")
        if not os.path.isdir(good_dir):
            raise RuntimeError(f"Missing folder: {good_dir}")
        for fn in sorted(os.listdir(good_dir)):
            if fn.lower().endswith(".png"):
                self.samples.append((os.path.join(good_dir, fn), 0))

        # 2) defects: either train/defect or fallback to test/*
        defect_dir = os.path.join(data_root, "train", "defect")
        if os.path.isdir(defect_dir):
            for fn in sorted(os.listdir(defect_dir)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(defect_dir, fn), 1))
        else:
            # fall back to test folder
            test_base = os.path.join(data_root, "test")
            if not os.path.isdir(test_base):
                raise RuntimeError(f"Missing test folder for fallback defects: {test_base}")
            for cls in sorted(os.listdir(test_base)):
                if cls == "good": 
                    continue
                cls_dir = os.path.join(test_base, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fn in sorted(os.listdir(cls_dir)):
                    if fn.lower().endswith(".png"):
                        self.samples.append((os.path.join(cls_dir, fn), 1))

        if not self.samples:
            raise RuntimeError("No training samples found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

# ─── Test Dataset ───────────────────────────────────────────────────────
class TestDataset(Dataset):
    """Loads test/good →0 and all other test/* →1"""
    def __init__(self, data_root, transform=None):
        self.transform = transform
        self.samples = []
        base = os.path.join(data_root, "test")
        if not os.path.isdir(base):
            raise RuntimeError(f"Missing test folder: {base}")
        for cls in sorted(os.listdir(base)):
            cls_dir = os.path.join(base, cls)
            if not os.path.isdir(cls_dir):
                continue
            lbl = 0 if cls == "good" else 1
            for fn in sorted(os.listdir(cls_dir)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fn), lbl))
        if not self.samples:
            raise RuntimeError("No test samples found!")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

# ─── Episode samplers ───────────────────────────────────────────────────
def sample_two_class_episode(ds, idxs_g, idxs_d, n_shot):
    """Support: n_shot good + n_shot defect (labels 0/1)."""
    good = random.sample(idxs_g, n_shot)
    bad  = random.sample(idxs_d, n_shot)
    idxs = good + bad
    imgs = torch.stack([ds[i][0] for i in idxs])
    lbls = torch.tensor([0]*n_shot + [1]*n_shot)
    return imgs, lbls

def sample_one_class_episode(ds, idxs_g, n_shot):
    """Support: n_shot good only (all-labels=0)."""
    good = random.sample(idxs_g, n_shot)
    imgs = torch.stack([ds[i][0] for i in good])
    lbls = torch.zeros(n_shot, dtype=torch.long)
    return imgs, lbls

def sample_test_episode(ds, idx_map, n_query):
    """Query: n_query good + n_query defect."""
    goods = random.sample(idx_map[0], n_query)
    bads  = random.sample(idx_map[1], n_query)
    idxs  = goods + bads
    imgs, lbls = zip(*(ds[i] for i in idxs))
    return torch.stack(imgs), torch.tensor(lbls)
    
def sample_good_episode(ds, idxs, n_shot, n_query):
    """
    Randomly sample n_shot + n_query “good” examples from ds at indices idxs.
    Returns:
      sup:  Tensor of shape [n_shot, 3, H, W]
      qry:  Tensor of shape [n_query, 3, H, W]
    """
    total     = n_shot + n_query
    pick      = random.sample(idxs, total)
    sup_idxs  = pick[:n_shot]
    qry_idxs  = pick[n_shot:]
    sup_imgs  = torch.stack([ds[i][0] for i in sup_idxs])
    qry_imgs  = torch.stack([ds[i][0] for i in qry_idxs])
    return sup_imgs, qry_imgs

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    freeze_support()
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      required=True)
    p.add_argument("--sam_checkpoint", required=True)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--n_shot",         type=int, default=5)
    p.add_argument("--n_query",        type=int, default=5)
    p.add_argument("--train_episodes", type=int, default=100)
    p.add_argument("--inner_steps",    type=int, default=5)
    p.add_argument("--eval_episodes",  type=int, default=100)
    args = p.parse_args()

    # reproducibility + device
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device(args.device)

    # transforms
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466,0.4578275,0.40821073],
            std =[0.26862954,0.26130258,0.27577711],
        ),
    ])

    # datasets
    train_ds = TrainDataset(args.data_root, transform=transform)
    test_ds  = TestDataset(args.data_root,  transform=transform)

    # split train indices
    idxs_g = [i for i,(_,lbl) in enumerate(train_ds.samples) if lbl==0]
    idxs_d = [i for i,(_,lbl) in enumerate(train_ds.samples) if lbl==1]
    two_class = len(idxs_d) > 0

    # build test index map
    test_map = {0:[],1:[]}
    for i,(_,lbl) in enumerate(test_ds.samples):
        test_map[lbl].append(i)

    print(f"# train good={len(idxs_g)}, defect={len(idxs_d)}")
    print(f"# test  good={len(test_map[0])}, defect={len(test_map[1])}")
    print("→ meta-training in", "2-class" if two_class else "1-class", "mode")

    # load & freeze SAM
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam = sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False
    C = sam.image_encoder.neck[0].out_channels

    # prompt parameters
    slow  = nn.Parameter(torch.randn(4, C, device=device)*0.02)
    fast  = nn.Parameter(torch.randn(2, C, device=device)*0.02)
    proto = nn.Parameter(torch.randn(1, C, device=device)*0.02)

    # optimizers + loss
    opt_f  = optim.SGD([fast],        lr=1e-2)
    opt_sp = optim.SGD([slow, proto], lr=1e-3)
    bceloss= nn.BCEWithLogitsLoss()

    # AMP
    scaler   = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    # helpers
    def encode(x):
        fmap = sam.image_encoder(x)      # [B,C,64,64]
        return fmap.mean([2,3])         # [B,C]

    def make_prompt(f):
        return proto + slow.mean(0,True) + f.mean(0,True)  # [1,C]

    # ─── Meta-training ────────────────────────────────────────────────────
    for ep in range(1, args.train_episodes+1):
        # sample support
        if two_class:
            sup_imgs, sup_lbls = sample_two_class_episode(
                train_ds, idxs_g, idxs_d, args.n_shot
            )
        else:
            sup_imgs, sup_lbls = sample_one_class_episode(
                train_ds, idxs_g, args.n_shot
            )
        sup_imgs = sup_imgs.to(device)
        sup_lbls = sup_lbls.float().to(device)

        # inner-loop: adapt fast only
        fast.data.zero_()
        for _ in range(args.inner_steps):
            opt_f.zero_grad()
            with autocast():
                feats = encode(sup_imgs)             # [B,C]
                prm   = make_prompt(fast)            # [1,C]
                dist  = (feats - prm).pow(2).sum(1).sqrt()
                loss  = bceloss(dist, sup_lbls)
            scaler.scale(loss).backward()
            scaler.step(opt_f)
            scaler.update()

        # outer-loop: update slow + proto
        fast.requires_grad_(False)
        opt_sp.zero_grad()
        with autocast():
            feats = encode(sup_imgs)
            prm   = make_prompt(fast)
            dist  = (feats - prm).pow(2).sum(1).sqrt()
            loss_q= bceloss(dist, sup_lbls)
        scaler.scale(loss_q).backward()
        scaler.step(opt_sp)
        scaler.update()
        fast.requires_grad_(True)

        print(f"[Train {ep:03d}/{args.train_episodes}] loss={loss_q.item():.4f}")

    # ─── Few-shot evaluation ───────────────────────────────────────────────
    print("\n=== Meta-training done; starting eval ===\n")
    accs, aucs = [], []

    for _ in tqdm(range(args.eval_episodes), desc="Eval"):
        # adapt on good only
        sup, _ = sample_one_class_episode(train_ds, idxs_g, args.n_shot)
        sup = sup.to(device)

        fast_eval = fast.clone().detach().requires_grad_(True)
        opt_fe    = optim.SGD([fast_eval], lr=1e-2)

        for _ in range(args.inner_steps):
            opt_fe.zero_grad()
            with autocast():
                feats = encode(sup)
                prm   = make_prompt(fast_eval)
                dist  = (feats - prm).pow(2).sum(1).sqrt()
                loss  = bceloss(dist, torch.zeros_like(dist))
            scaler.scale(loss).backward()
            scaler.step(opt_fe)
            scaler.update()

        # threshold = max support distance
        with torch.no_grad():
            feats = encode(sup)
            prm   = make_prompt(fast_eval).squeeze(0)
            d_sup = (feats - prm).pow(2).sum(1).sqrt()
            thr   = d_sup.max().item()

        # query
        qry_imgs, qry_lbls = sample_test_episode(test_ds, test_map, args.n_query)
        qry, lbls = qry_imgs.to(device), qry_lbls.numpy()
        with autocast(), torch.no_grad():
            feats = encode(qry)
            prm   = make_prompt(fast_eval).squeeze(0)
            d_q   = (feats - prm).pow(2).sum(1).sqrt().cpu().numpy()

        preds = (d_q > thr).astype(int)
        accs.append((preds == lbls).mean())
        aucs.append(roc_auc_score(lbls, d_q))

    print(f"\nAccuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"AUROC:    {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__=="__main__":
    main()
