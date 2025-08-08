# #!/usr/bin/env python3
# """
# oneclass_capsule.py

# One-class anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
# Meta-train only on good images; evaluate few-shot on test/good vs. test/defects.
# """

# import os
# import random
# import argparse
# import numpy as np
# from multiprocessing import freeze_support

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image, ImageFile

# # allow truncated PNGs
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from segment_anything import sam_model_registry

# # --- Dataset definitions ----------------------------------------------

# class GoodTrainDataset(Dataset):
#     """Good‐only training set: train/good/*.png → label 0"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         folder = os.path.join(data_root, "train", "good")
#         if not os.path.isdir(folder):
#             raise RuntimeError(f"Expected folder: {folder}")
#         self.samples = [
#             os.path.join(folder, fn)
#             for fn in sorted(os.listdir(folder))
#             if fn.lower().endswith(".png")
#         ]
#         if not self.samples:
#             raise RuntimeError("No training samples found under train/good/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         img = Image.open(self.samples[i]).convert("RGB")
#         if self.transform: img = self.transform(img)
#         return img, 0


# class TestDataset(Dataset):
#     """Test set: both good (0) and defects (1) from test/"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         base = os.path.join(data_root, "test")
#         if not os.path.isdir(base):
#             raise RuntimeError(f"Expected folder: {base}")
#         self.samples = []
#         for cls in sorted(os.listdir(base)):
#             folder = os.path.join(base, cls)
#             if not os.path.isdir(folder): continue
#             lbl = 0 if cls == "good" else 1
#             for fn in sorted(os.listdir(folder)):
#                 if fn.lower().endswith(".png"):
#                     self.samples.append((os.path.join(folder, fn), lbl))
#         if not self.samples:
#             raise RuntimeError("No test samples found under test/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         path, lbl = self.samples[i]
#         img = Image.open(path).convert("RGB")
#         if self.transform: img = self.transform(img)
#         return img, lbl

# # --- Episode samplers -----------------------------------------------

# def sample_good_episode(ds, idxs, n_shot, n_query):
#     """
#     Support/query for good class only.
#     Returns support_imgs, query_imgs (Tensors), both label=0.
#     """
#     total = n_shot + n_query
#     if len(idxs) < total:
#         raise RuntimeError(f"Need {total} good images, have {len(idxs)}")
#     pick = random.sample(idxs, total)
#     sup_idx = pick[:n_shot]
#     qry_idx = pick[n_shot:]
#     sup = [ds[i][0] for i in sup_idx]
#     qry = [ds[i][0] for i in qry_idx]
#     return torch.stack(sup), torch.stack(qry)

# def sample_test_episode(ds, idx_map, n_query):
#     """
#     n_query good and n_query defect from test set.
#     Returns (imgs, labels) for query.
#     """
#     goods = random.sample(idx_map[0], n_query)
#     bads  = random.sample(idx_map[1], n_query)
#     idxs = goods + bads
#     imgs, lbls = zip(*(ds[i] for i in idxs))
#     return torch.stack(imgs), torch.tensor(lbls)

# # --- Main ----------------------------------------------------------------

# def main():
#     freeze_support()
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",      type=str, required=True)
#     p.add_argument("--sam_checkpoint", type=str, required=True)
#     p.add_argument("--device",         type=str, default="cpu")
#     p.add_argument("--n_shot",         type=int, default=5)
#     p.add_argument("--n_query",        type=int, default=5)
#     p.add_argument("--train_episodes", type=int, default=50)
#     p.add_argument("--inner_steps",    type=int, default=5)
#     p.add_argument("--eval_episodes",  type=int, default=100)
#     args = p.parse_args()

#     # reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     device = torch.device(args.device)

#     # transforms (match SAM)
#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466, 0.4578275, 0.40821073],
#             std =[0.26862954, 0.26130258, 0.27577711],
#         ),
#     ])

#     # datasets
#     train_ds = GoodTrainDataset(args.data_root, transform=transform)
#     test_ds  = TestDataset(args.data_root,   transform=transform)

#     # index maps
#     train_idxs = list(range(len(train_ds)))  # all good
#     test_map = {0: [], 1: []}
#     for i,(_,lbl) in enumerate(test_ds.samples):
#         test_map[lbl].append(i)

#     print(f"# train good = {len(train_idxs)}")
#     print(f"# test good = {len(test_map[0])}, defect = {len(test_map[1])}")

#     # load SAM
#     sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
#     sam = sam.to(device).eval()
#     C = sam.image_encoder.neck[0].out_channels

#     # prompts & prototype
#     slow  = nn.Parameter(torch.randn(4,  C, device=device)*0.02)
#     fast  = nn.Parameter(torch.randn(2,  C, device=device)*0.02)
#     proto = nn.Parameter(torch.randn(1,  C, device=device)*0.02)

#     opt_fast  = optim.SGD([fast],           lr=1e-2)
#     opt_slowp = optim.SGD([slow, proto],   lr=1e-3)
#     mse       = nn.MSELoss()

#     def encode(x):
#         with torch.no_grad():
#             fmap = sam.image_encoder(x.to(device))
#             return fmap.mean([2,3])        # → [B,C] on cuda:0

#     def make_prompt():
#         return proto \
#              + slow.mean(0, keepdim=True) \
#              + fast.mean(0, keepdim=True)  # also on cuda:0

#     # --- meta-train on GOOD only ---
#     for ep in range(1, args.train_episodes+1):
#         sup, qry = sample_good_episode(train_ds, train_idxs,
#                                        args.n_shot, args.n_query)
#         sup, qry = sup.to(device), qry.to(device)
#         # inner adapt fast
#         fast.data.zero_()
#         for _ in range(args.inner_steps):
#             opt_fast.zero_grad()
#             loss = mse(encode(sup), make_prompt().expand_as(encode(sup)))
#             loss.backward()
#             opt_fast.step()
#         # outer update slow+proto
#         fast.requires_grad_(False)
#         opt_slowp.zero_grad()
#         loss_q = mse(encode(qry), make_prompt().expand_as(encode(qry)))
#         loss_q.backward()
#         opt_slowp.step()
#         fast.requires_grad_(True)

#         print(f"[Train epi {ep:02d}/{args.train_episodes}] loss: {loss_q.item():.4f}")

#     # --- few-shot eval on test/good vs. test/defects ---
#     accs = []
#     for _ in range(args.eval_episodes):
#         # support only good
#         sup, _ = sample_good_episode(train_ds, train_idxs,
#                                      args.n_shot, args.n_query)
#         sup = sup.to(device)
#         fast.data.zero_()
#         for _ in range(args.inner_steps):
#             opt_fast.zero_grad()
#             loss = mse(encode(sup), make_prompt().expand_as(encode(sup)))
#             loss.backward()
#             opt_fast.step()

#         # threshold = max dist on support
#         with torch.no_grad():
#             f_sup = encode(sup)
#             p     = make_prompt()                # [1,C]
#             d_sup = (f_sup - p).pow(2).sum(1).sqrt()
#             thr   = d_sup.max().item()

#         # query mix
#         qry_imgs, qry_lbls = sample_test_episode(test_ds, test_map,
#                                                  args.n_query)
#         qry_imgs = qry_imgs.to(device)
#         with torch.no_grad():
#             f_q = encode(qry_imgs)
#             d_q = (f_q - p).pow(2).sum(1).sqrt().cpu().numpy()
#             preds = (d_q > thr).astype(int)
#         accs.append((preds == qry_lbls.numpy()).mean())

#     print(f"\nFinal few-shot anomaly accuracy: "
#           f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

# if __name__=="__main__":
#     main()


#!/usr/bin/env python3
# """
# oneclass_capsule.py

# One-class anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
# Meta-train only on good images; evaluate few-shot on test/good vs. test/defects.
# Training runs on --device (cpu|cuda); evaluation always on CPU (same 1024×1024 inputs).
# """

# import os
# import random
# import argparse
# import numpy as np
# from multiprocessing import freeze_support

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image, ImageFile

# # allow truncated PNGs
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from segment_anything import sam_model_registry

# # --- Dataset definitions ----------------------------------------------

# class GoodTrainDataset(Dataset):
#     """Good‐only training set: train/good/*.png → label 0"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         folder = os.path.join(data_root, "train", "good")
#         if not os.path.isdir(folder):
#             raise RuntimeError(f"Expected folder: {folder}")
#         self.samples = [
#             os.path.join(folder, fn)
#             for fn in sorted(os.listdir(folder))
#             if fn.lower().endswith(".png")
#         ]
#         if not self.samples:
#             raise RuntimeError("No training samples found under train/good/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         img = Image.open(self.samples[i]).convert("RGB")
#         if self.transform: img = self.transform(img)
#         return img, 0


# class TestDataset(Dataset):
#     """Test set: both good (0) and defects (1) from test/"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         base = os.path.join(data_root, "test")
#         if not os.path.isdir(base):
#             raise RuntimeError(f"Expected folder: {base}")
#         self.samples = []
#         for cls in sorted(os.listdir(base)):
#             folder = os.path.join(base, cls)
#             if not os.path.isdir(folder): continue
#             lbl = 0 if cls == "good" else 1
#             for fn in sorted(os.listdir(folder)):
#                 if fn.lower().endswith(".png"):
#                     self.samples.append((os.path.join(folder, fn), lbl))
#         if not self.samples:
#             raise RuntimeError("No test samples found under test/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         path, lbl = self.samples[i]
#         img = Image.open(path).convert("RGB")
#         if self.transform: img = self.transform(img)
#         return img, lbl

# # --- Episode samplers -----------------------------------------------

# def sample_good_episode(ds, idxs, n_shot, n_query):
#     """
#     Support/query for good class only.
#     Returns support_imgs, query_imgs (Tensors), both label=0.
#     """
#     total = n_shot + n_query
#     if len(idxs) < total:
#         raise RuntimeError(f"Need {total} good images, have {len(idxs)}")
#     pick = random.sample(idxs, total)
#     sup_idxs, qry_idxs = pick[:n_shot], pick[n_shot:]
#     sup = torch.stack([ds[i][0] for i in sup_idxs])
#     qry = torch.stack([ds[i][0] for i in qry_idxs])
#     return sup, qry

# def sample_test_episode(ds, idx_map, n_query):
#     """
#     n_query good and n_query defect from test set.
#     Returns (imgs, labels) for query.
#     """
#     goods = random.sample(idx_map[0], n_query)
#     bads  = random.sample(idx_map[1], n_query)
#     idxs = goods + bads
#     imgs, lbls = zip(*(ds[i] for i in idxs))
#     return torch.stack(imgs), torch.tensor(lbls)

# # --- Main ----------------------------------------------------------------

# def main():
#     freeze_support()
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",      type=str, required=True)
#     p.add_argument("--sam_checkpoint", type=str, required=True)
#     p.add_argument("--device",         type=str, default="cpu",
#                    help="training device (cpu or cuda)")
#     p.add_argument("--n_shot",         type=int, default=5)
#     p.add_argument("--n_query",        type=int, default=5)
#     p.add_argument("--train_episodes", type=int, default=50)
#     p.add_argument("--inner_steps",    type=int, default=5)
#     p.add_argument("--eval_episodes",  type=int, default=100)
#     args = p.parse_args()

#     # reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     train_device = torch.device(args.device)

#     # 1) transforms — full 1024×1024 for both train & test
#     transform = transforms.Compose([
#         transforms.Resize((1024, 1024)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466, 0.4578275, 0.40821073],
#             std =[0.26862954, 0.26130258, 0.27577711],
#         ),
#     ])

#     # 2) Datasets
#     train_ds = GoodTrainDataset(args.data_root, transform=transform)
#     test_ds  = TestDataset(args.data_root,   transform=transform)

#     train_idxs = list(range(len(train_ds)))
#     test_map = {0: [], 1: []}
#     for i,(_,lbl) in enumerate(test_ds.samples):
#         test_map[lbl].append(i)

#     print(f"# train good = {len(train_idxs)}")
#     print(f"# test good = {len(test_map[0])}, defect = {len(test_map[1])}")

#     # 3) Load SAM → train_device
#     sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
#     sam = sam.to(train_device).eval()
#     C = sam.image_encoder.neck[0].out_channels

#     # 4) Prompt & prototype parameters on train_device
#     slow  = nn.Parameter(torch.randn(4,  C, device=train_device)*0.02)
#     fast  = nn.Parameter(torch.randn(2,  C, device=train_device)*0.02)
#     proto = nn.Parameter(torch.randn(1,  C, device=train_device)*0.02)

#     opt_fast  = optim.SGD([fast],           lr=1e-2)
#     opt_slowp = optim.SGD([slow, proto],   lr=1e-3)
#     mse       = nn.MSELoss()

#     # helper: encode on train_device
#     def encode_train(x):
#         with torch.no_grad():
#             fmap = sam.image_encoder(x.to(train_device))
#             return fmap.mean([2,3])

#     def make_prompt_train():
#         return proto \
#              + slow.mean(0, keepdim=True) \
#              + fast.mean(0, keepdim=True)

#     # 5) Meta-train on GOOD only
#     for ep in range(1, args.train_episodes+1):
#         sup, qry = sample_good_episode(train_ds, train_idxs,
#                                        args.n_shot, args.n_query)
#         sup, qry = sup.to(train_device), qry.to(train_device)

#         # Inner loop: adapt fast
#         fast.data.zero_()
#         for _ in range(args.inner_steps):
#             opt_fast.zero_grad()
#             loss = mse(encode_train(sup),
#                        make_prompt_train().expand_as(encode_train(sup)))
#             loss.backward()
#             opt_fast.step()

#         # Outer loop: update slow + proto
#         fast.requires_grad_(False)
#         opt_slowp.zero_grad()
#         loss_q = mse(encode_train(qry),
#                      make_prompt_train().expand_as(encode_train(qry)))
#         loss_q.backward()
#         opt_slowp.step()
#         fast.requires_grad_(True)

#         print(f"[Train epi {ep:02d}/{args.train_episodes}] loss: {loss_q.item():.4f}")

#     # 6) Move EVERYTHING to CPU for evaluation
#     print("\n=== Meta-training done; moving model & prompts to CPU for eval ===\n")
#     sam_cpu   = sam.cpu()
#     slow_cpu  = slow.detach().cpu()
#     fast_cpu  = fast.detach().cpu()
#     proto_cpu = proto.detach().cpu()

#     # helper: encode & prompt on CPU
#     def encode_eval(x):
#         with torch.no_grad():
#             fmap = sam_cpu.image_encoder(x.cpu())
#             return fmap.mean([2,3])
#     def make_prompt_eval():
#         return proto_cpu \
#              + slow_cpu.mean(0, keepdim=True) \
#              + fast_cpu.mean(0, keepdim=True)

#     # 7) Few-shot anomaly eval
#     accs = []
#     for _ in range(args.eval_episodes):
#         # adapt fast_prompt on CPU with support of only good
#         sup, _ = sample_good_episode(train_ds, train_idxs,
#                                      args.n_shot, args.n_query)
#         sup = sup.cpu()
#         # reset fast on CPU
#         fast_val = fast_cpu.clone().requires_grad_(True)
#         opt_fval = optim.SGD([fast_val], lr=1e-2)

#         for _ in range(args.inner_steps):
#             opt_fval.zero_grad()
#             loss = mse(encode_eval(sup),
#                        (proto_cpu + slow_cpu.mean(0,keepdim=True) + fast_val.mean(0,keepdim=True))
#                        .expand_as(encode_eval(sup)))
#             loss.backward()
#             opt_fval.step()

#         # threshold by max support‐distance
#         with torch.no_grad():
#             f_sup = encode_eval(sup)                              # [n_s+n_q, C]
#             p     = (proto_cpu + slow_cpu.mean(0,keepdim=True)
#                      + fast_val.mean(0,keepdim=True)).squeeze(0)
#             d_sup = (f_sup - p).pow(2).sum(1).sqrt()
#             thr   = d_sup.max().item()

#         # test mix
#         qry_imgs, qry_lbls = sample_test_episode(test_ds, test_map,
#                                                  args.n_query)
#         f_q = encode_eval(qry_imgs)                              # [2*n_q, C]
#         p   = p.unsqueeze(0)
#         d_q = (f_q - p).pow(2).sum(1).sqrt().numpy()
#         preds = (d_q > thr).astype(int)
#         accs.append((preds == qry_lbls.numpy()).mean())

#     print(f"\nFinal few-shot anomaly accuracy: "
#           f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")

# if __name__=="__main__":
#     main()

#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
# """
# oneclass_capsule_gpu.py

# One‐class anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
# Meta‐train only on good images; evaluate few‐shot on test/good vs. test/defects.
# Everything (train + eval) runs on the same device (GPU or CPU).
# """

# import os
# import random
# import argparse
# import numpy as np
# from multiprocessing import freeze_support

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image, ImageFile
# from tqdm import tqdm

# # allow truncated PNGs
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# from segment_anything import sam_model_registry

# # --- Dataset definitions ----------------------------------------------
# class GoodTrainDataset(Dataset):
#     """Good‐only training set: train/good/*.png → label 0"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         folder = os.path.join(data_root, "train", "good")
#         if not os.path.isdir(folder):
#             raise RuntimeError(f"Expected folder: {folder}")
#         self.samples = [
#             os.path.join(folder, fn)
#             for fn in sorted(os.listdir(folder))
#             if fn.lower().endswith(".png")
#         ]
#         if not self.samples:
#             raise RuntimeError("No training samples found under train/good/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         img = Image.open(self.samples[i]).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, 0

# class TestDataset(Dataset):
#     """Test set: both good (0) and defects (1) from test/"""
#     def __init__(self, data_root, transform=None):
#         self.transform = transform
#         base = os.path.join(data_root, "test")
#         if not os.path.isdir(base):
#             raise RuntimeError(f"Expected folder: {base}")
#         self.samples = []
#         for cls in sorted(os.listdir(base)):
#             folder = os.path.join(base, cls)
#             if not os.path.isdir(folder):
#                 continue
#             lbl = 0 if cls=="good" else 1
#             for fn in sorted(os.listdir(folder)):
#                 if fn.lower().endswith(".png"):
#                     self.samples.append((os.path.join(folder, fn), lbl))
#         if not self.samples:
#             raise RuntimeError("No test samples found under test/")
#     def __len__(self): return len(self.samples)
#     def __getitem__(self, i):
#         path, lbl = self.samples[i]
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, lbl

# # --- Episode samplers -----------------------------------------------
# def sample_good_episode(ds, idxs, n_shot, n_query):
#     total = n_shot + n_query
#     if len(idxs) < total:
#         raise RuntimeError(f"Need {total} good images, have {len(idxs)}")
#     pick = random.sample(idxs, total)
#     sup_idx, qry_idx = pick[:n_shot], pick[n_shot:]
#     sup = torch.stack([ds[i][0] for i in sup_idx])
#     qry = torch.stack([ds[i][0] for i in qry_idx])
#     return sup, qry

# def sample_test_episode(ds, idx_map, n_query):
#     goods = random.sample(idx_map[0], n_query)
#     bads  = random.sample(idx_map[1], n_query)
#     idxs  = goods + bads
#     imgs, lbls = zip(*(ds[i] for i in idxs))
#     return torch.stack(imgs), torch.tensor(lbls)

# # --- Main ----------------------------------------------------------------
# def main():
#     freeze_support()
#     p = argparse.ArgumentParser()
#     p.add_argument("--data_root",      type=str, required=True)
#     p.add_argument("--sam_checkpoint", type=str, required=True)
#     p.add_argument("--device",         type=str, default="cpu",
#                    help="train/eval device: cpu or cuda")
#     p.add_argument("--n_shot",         type=int, default=5)
#     p.add_argument("--n_query",        type=int, default=5)
#     p.add_argument("--train_episodes", type=int, default=50)
#     p.add_argument("--inner_steps",    type=int, default=5)
#     p.add_argument("--eval_episodes",  type=int, default=100)
#     args = p.parse_args()

#     # reproducibility
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     device = torch.device(args.device)

#     # 1) transforms — full 1024×1024 for both train & test
#     transform = transforms.Compose([
#         transforms.Resize((1024, 1024)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466, 0.4578275, 0.40821073],
#             std =[0.26862954, 0.26130258, 0.27577711],
#         ),
#     ])

#     # 2) Datasets & index maps
#     train_ds = GoodTrainDataset(args.data_root, transform=transform)
#     test_ds  = TestDataset(args.data_root,  transform=transform)

#     train_idxs = list(range(len(train_ds)))
#     test_map = {0: [], 1: []}
#     for i, (_, lbl) in enumerate(test_ds.samples):
#         test_map[lbl].append(i)

#     print(f"# train good = {len(train_idxs)}")
#     print(f"# test good = {len(test_map[0])}, defect = {len(test_map[1])}")

#     # 3) Load & freeze SAM on device
#     sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
#     sam = sam.to(device).eval()
#     # freeze all SAM weights
#     for p in sam.parameters():
#         p.requires_grad = False

#     C = sam.image_encoder.neck[0].out_channels

#     # 4) Prompt & prototype parameters (on device)
#     slow  = nn.Parameter(torch.randn(4,  C, device=device)*0.02)
#     fast  = nn.Parameter(torch.randn(2,  C, device=device)*0.02)
#     proto = nn.Parameter(torch.randn(1,  C, device=device)*0.02)

#     opt_fast  = optim.SGD([fast],         lr=1e-2)
#     opt_slowp = optim.SGD([slow, proto],  lr=1e-3)
#     mse       = nn.MSELoss()

#     # helpers on device
#     def encode_dev(x):
#         fmap = sam.image_encoder(x.to(device))   # [B,C,64,64]
#         return fmap.mean([2,3])                  # [B,C]

#     def make_prompt_dev(fast_tensor):
#         # fast_tensor is either `fast` during train or `fast_eval` during eval
#         return proto + slow.mean(0,keepdim=True) + fast_tensor.mean(0,keepdim=True)

#     # 5) Meta‐train on GOOD only
#     for ep in range(1, args.train_episodes+1):
#         sup, qry = sample_good_episode(train_ds, train_idxs,
#                                        args.n_shot, args.n_query)
#         sup, qry = sup.to(device), qry.to(device)

#         # inner‐loop adapt only `fast`
#         fast.data.zero_()
#         for _ in range(args.inner_steps):
#             opt_fast.zero_grad()
#             feat_sup = encode_dev(sup)                          # [n_s+n_q, C]
#             prompt   = make_prompt_dev(fast)                    # [1,C]
#             loss = mse(feat_sup, prompt.expand_as(feat_sup))
#             loss.backward()
#             opt_fast.step()

#         # outer‐loop update slow + proto
#         fast.requires_grad_(False)
#         opt_slowp.zero_grad()
#         feat_q  = encode_dev(qry)
#         prompt  = make_prompt_dev(fast)
#         loss_q  = mse(feat_q, prompt.expand_as(feat_q))
#         loss_q.backward()
#         opt_slowp.step()
#         fast.requires_grad_(True)

#         print(f"[Train epi {ep:02d}/{args.train_episodes}] loss: {loss_q.item():.4f}")

#     # 6) Few‐shot eval on SAME device, with live tqdm bar
#     print("\n=== Meta‐training done; few‐shot eval on device ===\n")
#     accs = []
#     for _epi in tqdm(range(args.eval_episodes), desc="Eval episodes"):
#         # support (only good)
#         sup, _ = sample_good_episode(train_ds, train_idxs,
#                                      args.n_shot, args.n_query)
#         sup = sup.to(device)

#         # reset a *new* fast tensor for this episode
#         fast_eval = fast.clone().detach().requires_grad_(True)
#         opt_feval = optim.SGD([fast_eval], lr=1e-2)

#         # inner‐loop on SUP only
#         for _ in range(args.inner_steps):
#             opt_feval.zero_grad()
#             feat_sup = encode_dev(sup)
#             prompt   = make_prompt_dev(fast_eval)
#             loss     = mse(feat_sup, prompt.expand_as(feat_sup))
#             loss.backward()
#             opt_feval.step()

#         # threshold = max distance on SUP
#         with torch.no_grad():
#             feat_sup = encode_dev(sup)
#             prompt   = make_prompt_dev(fast_eval).squeeze(0)  # [C]
#             d_sup    = (feat_sup - prompt).pow(2).sum(1).sqrt()
#             thr      = d_sup.max().item()

#         # sample & classify QUERY mix
#         qry_imgs, qry_lbls = sample_test_episode(test_ds, test_map,
#                                                  args.n_query)
#         qry = qry_imgs.to(device)
#         with torch.no_grad():
#             feat_q = encode_dev(qry)
#             d_q    = (feat_q - prompt.unsqueeze(0)).pow(2).sum(1).sqrt().cpu().numpy()
#             preds  = (d_q > thr).astype(int)

#         accs.append((preds == qry_lbls.numpy()).mean())

#     mean_acc, std_acc = np.mean(accs), np.std(accs)
#     print(f"\nFinal few-shot anomaly accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

# if __name__=="__main__":
#     main()

#!/usr/bin/env python3
#!/usr/bin/env python3

# Final few-shot anomaly accuracy: 0.5660 ± 0.1041
# Final few-shot anomaly AUROC:     0.6032 ± 0.1888
"""
oneclass_capsule_gpu_amp.py

One-class anomaly detection on MVTec “capsule” with SAM + fast/slow prompt tuning.
Meta-train only on good images; evaluate few-shot on test/good vs. test/defects.
Everything (train + eval) runs on the same device (GPU or CPU), using mixed precision.
Reports both accuracy and AUROC.
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

# --- Dataset definitions ----------------------------------------------
class GoodTrainDataset(Dataset):
    """Good-only training set: train/good/*.png → label 0"""
    def __init__(self, data_root, transform=None):
        self.transform = transform
        folder = os.path.join(data_root, "train", "good")
        if not os.path.isdir(folder):
            raise RuntimeError(f"Expected folder: {folder}")
        self.samples = [
            os.path.join(folder, fn)
            for fn in sorted(os.listdir(folder))
            if fn.lower().endswith(".png")
        ]
        if not self.samples:
            raise RuntimeError("No training samples found under train/good/")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

class TestDataset(Dataset):
    """Test set: both good (0) and defects (1) from test/"""
    def __init__(self, data_root, transform=None):
        self.transform = transform
        base = os.path.join(data_root, "test")
        if not os.path.isdir(base):
            raise RuntimeError(f"Expected folder: {base}")
        self.samples = []
        for cls in sorted(os.listdir(base)):
            folder = os.path.join(base, cls)
            if not os.path.isdir(folder):
                continue
            lbl = 0 if cls == "good" else 1
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(".png"):
                    self.samples.append((os.path.join(folder, fn), lbl))
        if not self.samples:
            raise RuntimeError("No test samples found under test/")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl

# --- Episode samplers -----------------------------------------------
def sample_good_episode(ds, idxs, n_shot, n_query):
    total = n_shot + n_query
    if len(idxs) < total:
        raise RuntimeError(f"Need {total} good images, have {len(idxs)}")
    pick = random.sample(idxs, total)
    sup_idx, qry_idx = pick[:n_shot], pick[n_shot:]
    sup = torch.stack([ds[i][0] for i in sup_idx])
    qry = torch.stack([ds[i][0] for i in qry_idx])
    return sup, qry

def sample_test_episode(ds, idx_map, n_query):
    goods = random.sample(idx_map[0], n_query)
    bads  = random.sample(idx_map[1], n_query)
    idxs  = goods + bads
    imgs, lbls = zip(*(ds[i] for i in idxs))
    return torch.stack(imgs), torch.tensor(lbls)

# --- Main ----------------------------------------------------------------
def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",      type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--device",         type=str, default="cpu",
                        help="train/eval device: cpu or cuda")
    parser.add_argument("--n_shot",         type=int, default=5)
    parser.add_argument("--n_query",        type=int, default=5)
    parser.add_argument("--train_episodes", type=int, default=50)
    parser.add_argument("--inner_steps",    type=int, default=5)
    parser.add_argument("--eval_episodes",  type=int, default=100)
    args = parser.parse_args()

    # reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)

    # 1) transforms — full 1024×1024 for both train & test
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # 2) Datasets & index maps
    train_ds = GoodTrainDataset(args.data_root, transform=transform)
    test_ds  = TestDataset(args.data_root,  transform=transform)

    train_idxs = list(range(len(train_ds)))
    test_map = {0: [], 1: []}
    for i, (_, lbl) in enumerate(test_ds.samples):
        test_map[lbl].append(i)

    print(f"# train good = {len(train_idxs)}")
    print(f"# test good = {len(test_map[0])}, defect = {len(test_map[1])}")

    # 3) Load & freeze SAM on device
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam = sam.to(device).eval()
    for p in sam.parameters():
        p.requires_grad = False

    C = sam.image_encoder.neck[0].out_channels

    # 4) Prompt & prototype parameters (on device)
    slow  = nn.Parameter(torch.randn(4,  C, device=device)*0.02)
    fast  = nn.Parameter(torch.randn(2,  C, device=device)*0.02)
    proto = nn.Parameter(torch.randn(1,  C, device=device)*0.02)

    opt_fast  = optim.SGD([fast],         lr=1e-2)
    opt_slowp = optim.SGD([slow, proto],  lr=1e-3)
    mse       = nn.MSELoss()

    # 5) AMP setup
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    # helpers on device
    def encode_dev(x):
        fmap = sam.image_encoder(x.to(device))   # [B,C,64,64]
        return fmap.mean([2,3])                  # [B,C]

    def make_prompt_dev(fast_tensor):
        return proto + slow.mean(0,keepdim=True) + fast_tensor.mean(0,keepdim=True)

    # 6) Meta-train on GOOD only with AMP
    for ep in range(1, args.train_episodes+1):
        sup, qry = sample_good_episode(train_ds, train_idxs,
                                       args.n_shot, args.n_query)
        sup, qry = sup.to(device), qry.to(device)

        # inner-loop adapt only `fast`
        fast.data.zero_()
        for _ in range(args.inner_steps):
            opt_fast.zero_grad()
            with autocast():
                feat_sup = encode_dev(sup)                          # [n_s+n_q, C]
                prompt   = make_prompt_dev(fast)                    # [1,C]
                loss     = mse(feat_sup, prompt.expand_as(feat_sup))
            scaler.scale(loss).backward()
            scaler.step(opt_fast)
            scaler.update()

        # outer-loop update slow + proto
        fast.requires_grad_(False)
        opt_slowp.zero_grad()
        with autocast():
            feat_q = encode_dev(qry)
            prompt = make_prompt_dev(fast)
            loss_q = mse(feat_q, prompt.expand_as(feat_q))
        scaler.scale(loss_q).backward()
        scaler.step(opt_slowp)
        scaler.update()
        fast.requires_grad_(True)

        print(f"[Train epi {ep:02d}/{args.train_episodes}] loss: {loss_q.item():.4f}")

    # 7) Few-shot eval on SAME device, with tqdm bar, AMP, accuracy & AUROC
    print("\n=== Meta-training done; few-shot eval on device ===\n")
    accs = []
    aucs = []
    for _ in tqdm(range(args.eval_episodes), desc="Eval episodes"):
        # support
        sup, _ = sample_good_episode(train_ds, train_idxs,
                                     args.n_shot, args.n_query)
        sup = sup.to(device)

        # reset a new fast tensor for this episode
        fast_eval = fast.clone().detach().requires_grad_(True)
        opt_feval = optim.SGD([fast_eval], lr=1e-2)

        # inner-loop on SUP only
        for _ in range(args.inner_steps):
            opt_feval.zero_grad()
            with autocast():
                feat_sup = encode_dev(sup)
                prompt   = make_prompt_dev(fast_eval)
                loss     = mse(feat_sup, prompt.expand_as(feat_sup))
            scaler.scale(loss).backward()
            scaler.step(opt_feval)
            scaler.update()

        # threshold = max distance on SUP
        with torch.no_grad():
            feat_sup = encode_dev(sup)
            prompt   = make_prompt_dev(fast_eval).squeeze(0)  # [C]
            d_sup    = (feat_sup - prompt).pow(2).sum(1).sqrt()
            thr      = d_sup.max().item()

        # sample & classify QUERY mix
        qry_imgs, qry_lbls = sample_test_episode(test_ds, test_map,
                                                 args.n_query)
        qry = qry_imgs.to(device)
        with autocast(), torch.no_grad():
            feat_q = encode_dev(qry)
            d_q    = (feat_q - prompt.unsqueeze(0)).pow(2).sum(1).sqrt().cpu().numpy()
            preds  = (d_q > thr).astype(int)

        # compute AUROC and accuracy for this episode
        aucs.append(roc_auc_score(qry_lbls.numpy(), d_q))
        accs.append((preds == qry_lbls.numpy()).mean())

    # summarize
    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    print(f"\nFinal few-shot anomaly accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Final few-shot anomaly AUROC:     {mean_auc:.4f} ± {std_auc:.4f}")

if __name__=="__main__":
    main()
