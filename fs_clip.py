#!/usr/bin/env python3
"""
fast_slow_openclip_prompt_tuning.py

Brain-Inspired Fast- and Slow-Update Prompt Tuning on Dummy Data with OpenCLIP

1) Load OpenCLIP (ViT-B/32) and freeze its weights.
2) Initialize slow & fast prompt vectors in the text embedding space.
3) Meta-train via inner (fast) & outer (slow) loops on dummy few-shot episodes.
4) Test-time adaptation and average few-shot accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import open_clip

# --- 0) Setup & Dummy Data ---
device = torch.device("cpu")       # CPU only to avoid macOS GPU/MPS issues
num_samples, num_classes = 20, 5
images = torch.randn(num_samples, 3, 224, 224, device=device)
labels = torch.randint(0, num_classes, (num_samples,), device=device)
class_names = [f"class {i}" for i in range(num_classes)]

# --- 1) Load OpenCLIP and Freeze ---
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
for p in model.parameters():
    p.requires_grad = False

# --- 2) Prompt Embeddings ---
# Determine projection dimension from text_projection parameter
# open_clip’s text_projection is a Parameter of shape [width, embed_dim]
embed_dim = model.text_projection.shape[1]  # e.g., 512

n_slow, n_fast = 4, 2
slow_prompts = nn.Parameter(torch.randn(n_slow, embed_dim) * 0.02)
fast_prompts = nn.Parameter(torch.randn(n_fast, embed_dim) * 0.02)

inner_opt = optim.SGD([fast_prompts], lr=1e-2)
outer_opt = optim.SGD([slow_prompts], lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def encode_image(batch):
    """Directly get OpenCLIP image features from raw pixel_values."""
    return model.encode_image(batch)

def encode_prompt(c):
    """
    Build and encode a prompt embedding for class index c:
    base text embedding + mean(slow_prompts) + mean(fast_prompts).
    """
    txt = class_names[c]
    tokens = tokenizer([txt]).to(device)
    base_emb = model.encode_text(tokens)  # [1, embed_dim]
    # Combine
    combo = base_emb + slow_prompts.mean(0, keepdim=True) + fast_prompts.mean(0, keepdim=True)
    return combo.squeeze(0)  # [embed_dim]

# --- 3) Meta-Training Loop ---
for outer in range(50):
    # 3-way, 1-shot episode
    ways = torch.randperm(num_classes)[:3]
    sup_idx, sup_lbls, qry_idx, qry_lbls = [], [], [], []
    for i, c in enumerate(ways):
        inds = (labels == c).nonzero(as_tuple=True)[0].tolist()
        sup_idx.append(inds[0]); sup_lbls.append(i)
        qry_idx.append(inds[1]); qry_lbls.append(i)
    sup_idx = torch.tensor(sup_idx, device=device)
    sup_lbls = torch.tensor(sup_lbls, device=device)
    qry_idx = torch.tensor(qry_idx, device=device)
    qry_lbls = torch.tensor(qry_lbls, device=device)

    # Inner loop: fast prompts
    fast_prompts.data.zero_()
    for _ in range(5):
        inner_opt.zero_grad()
        img_feats = encode_image(images[sup_idx])           # [3, dim]
        txt_feats = torch.stack([encode_prompt(int(c))      # [3, dim]
                                  for c in ways])
        logits = img_feats @ txt_feats.t()                   # [3,3]
        loss = loss_fn(logits, sup_lbls)
        loss.backward()
        inner_opt.step()

    # Outer loop: slow prompts
    fast_prompts.requires_grad_(False)
    outer_opt.zero_grad()
    img_feats_q = encode_image(images[qry_idx])
    txt_feats_q = torch.stack([encode_prompt(int(c)) for c in ways])
    logits_q = img_feats_q @ txt_feats_q.t()
    loss_q = loss_fn(logits_q, qry_lbls)
    loss_q.backward()
    outer_opt.step()
    fast_prompts.requires_grad_(True)

# --- 4) Test-Time Adaptation & Evaluation ---
fast_prompts.data.zero_()
for _ in range(5):
    inner_opt.zero_grad()
    img_feats = encode_image(images[sup_idx])
    txt_feats = torch.stack([encode_prompt(int(c)) for c in ways])
    logits = img_feats @ txt_feats.t()
    loss = loss_fn(logits, sup_lbls)
    loss.backward()
    inner_opt.step()

# Final few-shot accuracy
img_feats = encode_image(images[qry_idx])
txt_feats = torch.stack([encode_prompt(int(c)) for c in ways])
preds = (img_feats @ txt_feats.t()).argmax(dim=1)
acc = (preds == qry_lbls).float().mean()

print(f"Few-shot accuracy on dummy data: {acc.item():.4f}")