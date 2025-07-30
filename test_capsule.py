# #!/usr/bin/env python3
# import os, random, argparse
# import numpy as np
# from multiprocessing import freeze_support

# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
# from segment_anything import sam_model_registry

# class BinaryCapsuleDataset(Dataset):
#     """Binary: 0=good, 1=any defect."""
#     def __init__(self, data_root, split="test", transform=None):
#         self.transform = transform
#         self.samples = []

#         split_dir = os.path.join(data_root, split)
#         for cls in os.listdir(split_dir):
#             cls_dir = os.path.join(split_dir, cls)
#             # skip non-directories (e.g. .DS_Store)
#             if not os.path.isdir(cls_dir):
#                 continue
#             label = 0 if cls == "good" else 1
#             for fn in os.listdir(cls_dir):
#                 if fn.lower().endswith(".png"):
#                     self.samples.append((os.path.join(cls_dir, fn), label))

#     def __len__(self): 
#         return len(self.samples)

#     def __getitem__(self, i):
#         path, lbl = self.samples[i]
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, lbl


# def sample_binary_episode(ds, idx_map, n_shot, n_query):
#     sup, s_lbls, qry, q_lbls = [],[],[],[]
#     for cls in (0,1):
#         inds = idx_map[cls]
#         picked = random.sample(inds, n_shot+n_query)
#         for p in picked[:n_shot]:
#             img,_ = ds[p]; sup.append(img); s_lbls.append(cls)
#         for p in picked[n_shot:]:
#             img,_ = ds[p]; qry.append(img); q_lbls.append(cls)
#     return (torch.stack(sup), torch.tensor(s_lbls),
#             torch.stack(qry),torch.tensor(q_lbls))

# def main():
#     freeze_support()
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",     required=True)
#     p.add_argument("--sam_checkpoint",required=True)
#     p.add_argument("--device",   default="cuda")
#     p.add_argument("--n_shot",   type=int, default=5)
#     p.add_argument("--n_query",  type=int, default=5)
#     p.add_argument("--epochs",   type=int, default=30)
#     p.add_argument("--eps_per_epoch", type=int, default=100)
#     args = p.parse_args()

#     random.seed(42); np.random.seed(42); torch.manual_seed(42)
#     dev = torch.device(args.device)

#     # ── transforms w/ augmentation ─────────────────────────
#     transform = transforms.Compose([
#       transforms.Resize((1024,1024)),
#       transforms.RandomHorizontalFlip(),
#       transforms.RandomCrop(1024, padding=32),
#       transforms.ToTensor(),
#       transforms.Normalize(
#         mean=[.48145,.45783,.40821],
#         std =[.26863,.26130,.27578],
#       ),
#     ])

#     ds = BinaryCapsuleDataset(args.data_root, "test", transform)
#     idx_map = {0:[],1:[]}
#     for i,(_,lbl) in enumerate(ds.samples): idx_map[lbl].append(i)

#     # ── load SAM ─────────────────────────────────────────────
#     sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint).to(dev).eval()
#     C = sam.image_encoder.neck[0].out_channels

#     # ── prompt & proto vars ──────────────────────────────────
#     slow   = nn.Parameter(torch.randn(4, C)*0.02)
#     fast   = nn.Parameter(torch.randn(2, C)*0.02)
#     proto  = nn.Parameter(torch.randn(2, C)*0.02)
#     opt_f  = optim.SGD([fast],          lr=1e-2)
#     opt_sp = optim.SGD([slow, proto],   lr=1e-3)
#     loss_fn = nn.CrossEntropyLoss()

#     def encode(x):
#         with torch.no_grad():
#             fmap = sam.image_encoder(x.to(dev))
#             return fmap.mean((2,3))

#     def allp():
#         return proto + slow.mean(0,True) + fast.mean(0,True)

#     # ── meta‐train ───────────────────────────────────────────
#     for ep in range(1, args.epochs+1):
#         total_loss=0.0
#         for _ in range(args.eps_per_epoch):
#             sup, s_lbls, qry, q_lbls = sample_binary_episode(
#               ds, idx_map, args.n_shot, args.n_query
#             )
#             fast.data.zero_()
#             # inner
#             for _ in range(10):   # more inner steps
#                 opt_f.zero_grad()
#                 l = loss_fn(encode(sup) @ allp().t(), s_lbls.to(dev))
#                 l.backward(); opt_f.step()
#             # outer
#             fast.requires_grad_(False)
#             opt_sp.zero_grad()
#             lq = loss_fn(encode(qry) @ allp().t(), q_lbls.to(dev))
#             lq.backward(); opt_sp.step()
#             fast.requires_grad_(True)
#             total_loss += lq.item()
#         print(f"Epoch {ep}/{args.epochs} — avg meta‐loss: {total_loss/args.eps_per_epoch:.4f}")

#     # ── evaluate over many episodes ─────────────────────────
#     accs=[]
#     for _ in range(100):
#         sup, s_lbls, qry, q_lbls = sample_binary_episode(
#           ds, idx_map, args.n_shot, args.n_query
#         )
#         fast.data.zero_()
#         for _ in range(10):
#             opt_f.zero_grad()
#             loss_fn(encode(sup) @ allp().t(), s_lbls.to(dev)).backward()
#             opt_f.step()
#         with torch.no_grad():
#             preds = (encode(qry) @ allp().t()).argmax(dim=1).cpu()
#             accs.append((preds==q_lbls).float().mean().item())
#     print(f"\nAvg few‐shot binary accuracy: {np.mean(accs):.4f}")

# if __name__=="__main__":
#     main()

#!/usr/bin/env python3
"""
Binary few-shot meta-learning on Capsule dataset (good vs defect) using SAM with fast/slow prompt tuning
"""
import os, random, argparse, time
import numpy as np
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from segment_anything import sam_model_registry

class BinaryCapsuleDataset(Dataset):
    """Binary classification: 0 = good, 1 = any defect"""
    def __init__(self, data_root, split="train", transform=None):
        self.transform = transform
        self.samples = []
        split_dir = os.path.join(data_root, split)
        for cls in os.listdir(split_dir):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = 0 if cls == "good" else 1
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fn), label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

def sample_binary_episode(dataset, idx_map, n_shot=5, n_query=5):
    """Draws one 2-way K-shot episode (good vs anomaly)."""
    sup, s_lbls, qry, q_lbls = [], [], [], []
    for cls in (0, 1):
        inds = idx_map[cls]
        picked = random.sample(inds, n_shot + n_query)
        for p in picked[:n_shot]:
            img, _ = dataset[p]
            sup.append(img); s_lbls.append(cls)
        for p in picked[n_shot:]:
            img, _ = dataset[p]
            qry.append(img); q_lbls.append(cls)
    return torch.stack(sup), torch.tensor(s_lbls), torch.stack(qry), torch.tensor(q_lbls)

if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(
        description="Binary few-shot SAM meta-learning: good vs defect"
    )
    parser.add_argument("--data_root",     type=str, required=True,
                        help="Path to dataset root (contains train/ and test/)")
    parser.add_argument("--sam_type",      type=str, default="vit_b",
                        help="SAM model type (e.g. vit_b)")
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                        help="Path to SAM checkpoint .pth file")
    parser.add_argument("--device",        type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--n_shot",        type=int, default=5,
                        help="Number of support examples per class")
    parser.add_argument("--n_query",       type=int, default=5,
                        help="Number of query examples per class")
    parser.add_argument("--epochs",        type=int, default=25,
                        help="Meta-training epochs")
    parser.add_argument("--episodes",      type=int, default=100,
                        help="Episodes per epoch")
    parser.add_argument("--inner_steps",   type=int, default=5,
                        help="Inner-loop (fast) gradient steps")
    parser.add_argument("--eval_episodes", type=int, default=100,
                        help="Number of test episodes for final evaluation")
    args = parser.parse_args()

    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)

    # transforms with augmentation
    transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(1024, pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # load train/test datasets
    ds_train = BinaryCapsuleDataset(args.data_root, split="train", transform=transform)
    ds_test  = BinaryCapsuleDataset(args.data_root, split="test",  transform=transform)

    # build index maps
    train_map = {0: [], 1: []}
    for i, (_, lbl) in enumerate(ds_train.samples):
        train_map[lbl].append(i)
    test_map  = {0: [], 1: []}
    for i, (_, lbl) in enumerate(ds_test.samples):
        test_map[lbl].append(i)

    # load SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(device).eval()
    C = sam.image_encoder.neck[0].out_channels

    # initialize fast/slow prompts and class prototypes
    slow  = nn.Parameter(torch.randn(4, C) * 0.02)
    fast  = nn.Parameter(torch.randn(2, C) * 0.02)
    proto = nn.Parameter(torch.randn(2, C) * 0.02)  # [good, anomaly]

    opt_fast  = optim.SGD([fast],          lr=1e-2)
    opt_slowp = optim.SGD([slow, proto],  lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()

    def encode_image(x):
        with torch.no_grad():
            fmap = sam.image_encoder(x.to(device))  # [B,C,64,64]
            return fmap.mean(dim=[2,3])            # [B,C]

    def all_prompts():
        avg_s = slow.mean(dim=0, keepdim=True)  # [1,C]
        avg_f = fast.mean(dim=0, keepdim=True)  # [1,C]
        return proto + avg_s + avg_f            # [2,C]

    # meta-training
    for epoch in range(1, args.epochs+1):
        tot_loss = 0.0
        for _ in range(args.episodes):
            sup, s_lbls, qry, q_lbls = sample_binary_episode(
                ds_train, train_map,
                n_shot=args.n_shot, n_query=args.n_query
            )
            sup, s_lbls = sup.to(device), s_lbls.to(device)
            qry, q_lbls = qry.to(device), q_lbls.to(device)

            # inner-loop: fast prompts
            fast.data.zero_()
            for _ in range(args.inner_steps):
                opt_fast.zero_grad()
                logits_sup = encode_image(sup) @ all_prompts().t()
                loss_sup   = loss_fn(logits_sup, s_lbls)
                loss_sup.backward()
                opt_fast.step()

            # outer-loop: slow prompts + prototypes
            fast.requires_grad_(False)
            opt_slowp.zero_grad()
            logits_q = encode_image(qry) @ all_prompts().t()
            loss_q   = loss_fn(logits_q, q_lbls)
            loss_q.backward()
            opt_slowp.step()
            fast.requires_grad_(True)

            tot_loss += loss_q.item()

        avg_loss = tot_loss / args.episodes
        print(f"Epoch {epoch:02d}/{args.epochs:02d} — avg meta-loss: {avg_loss:.4f}")

    # final evaluation on test set
    accs = []
    start = time.time()
    for _ in range(args.eval_episodes):
        sup, s_lbls, qry, q_lbls = sample_binary_episode(
            ds_test, test_map,
            n_shot=args.n_shot, n_query=args.n_query
        )
        sup, s_lbls = sup.to(device), s_lbls.to(device)
        qry, q_lbls = qry.to(device), q_lbls.to(device)

        fast.data.zero_()
        for _ in range(args.inner_steps):
            opt_fast.zero_grad()
            logits_sup = encode_image(sup) @ all_prompts().t()
            loss_sup   = loss_fn(logits_sup, s_lbls)
            loss_sup.backward()
            opt_fast.step()

        with torch.no_grad():
            preds = (encode_image(qry) @ all_prompts().t()).argmax(dim=1)
            accs.append((preds.cpu() == q_lbls.cpu()).float().mean().item())

    elapsed = time.time() - start
    accs = np.array(accs)
    print(f"\nTest on {args.eval_episodes} episodes (CPU) in {elapsed:.1f}s")
    print(f"Mean accuracy: {accs.mean():.4f} ± {accs.std():.4f}")

