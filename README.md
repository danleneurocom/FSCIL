 UCAD-Style Continual Anomaly Detection

This README explains how to run the provided **Fast and slow prompt tuning for continual anomaly detection** experiment end to end.

## 1. What this code does

The script trains and evaluates a **continual anomaly detection** pipeline on **MVTec AD**.

Main components in the code:
- **Frozen ViT-B/16** backbone
- **Fast / slow additive prompts** injected per transformer block
- **Inner / outer meta-training**
- **Per-task memory**:
  - `keys` from an early layer
  - `knowledge` from the final layer
- **Task-agnostic inference** by choosing the task with the **lowest image anomaly score**
- **Structure-aware contrastive learning (SCL)** using **SAM** (or SLIC fallback)
- **Continual evaluation** after each task on all seen categories

The default experiment runs on the 15 MVTec AD categories:
`bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper`

---

## 2. Recommended environment

- Python 3.10 or 3.11
- PyTorch with CUDA if available
- GPU strongly recommended

The script automatically uses:
- `cuda` if available
- otherwise `mps` on Apple Silicon
- otherwise `cpu`

---

## 3. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

python -m pip install --upgrade pip
```

Install PyTorch first using the command appropriate for your system from the official PyTorch install page.

Then install the required Python packages:

```bash
pip install torch torchvision numpy scikit-image scikit-learn tqdm Pillow
pip install opencv-python timm
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Or simply just install everything in ```requirement.txt```

If `segment-anything` is not installed or the SAM checkpoint path is missing, the code falls back to **SLIC** for structure segmentation.

---

## 4. Download the dataset

This project can be prepared with either **MVTec AD** or **VisA**. MVTec AD already matches the current loader directly. VisA needs one extra conversion step before it can be used with this exact script.

### 4.1 MVTec AD

The official MVTec AD site provides both the full dataset and per-category downloads. Each category contains `train`, `test`, and `ground_truth`, which is exactly the structure expected by your current `MVTecAD` loader. citeturn535417search2turn535417search6

#### Expected folder structure

After extraction, the root should look like this:

```txt
mvtec2d/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   ├── broken_small/
│   │   └── contamination/
│   └── ground_truth/
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
├── ...
└── zipper/
```

This matches what the loader expects:

```python
root/category/train/good/*
root/category/test/<defect_type>/*
root/category/ground_truth/<defect_type>/*
```

#### Rename / place the dataset

If your extracted folder is named something else, you can either:
- rename it to `mvtec2d`, or
- keep the original name and update `DATA_ROOT` in the script.

Example:

```python
DATA_ROOT = "/absolute/path/to/mvtec2d"
```

### 4.2 VisA

VisA is released by Amazon Science as the **Visual Anomaly** dataset with **10,821** images across **12** objects, with both image-level and pixel-level labels. Amazon Science also provides the official dataset page, and the `amazon-science/spot-diff` repository documents the download location and raw folder tree. citeturn535417search0turn535417search5turn535417search12

#### Raw VisA structure

The official VisA release uses a different layout from MVTec, roughly like:

```txt
VisA/
├── candle/
│   ├── Data/
│   │   ├── Images/
│   │   │   ├── Anomaly/
│   │   │   └── Normal/
│   │   └── Masks/
│   │       └── Anomaly/
│   └── image_anno.csv
├── capsules/
├── cashew/
├── chewinggum/
├── fryum/
├── macaroni1/
├── macaroni2/
├── pcb1/
├── pcb2/
├── pcb3/
├── pcb4/
└── pipe_fryum/
```

Because your current code expects the MVTec folder convention, VisA will **not** run directly with the existing `MVTecAD` class. You need to convert each VisA category into a MVTec-like layout first.

#### Recommended converted VisA structure

Convert VisA into:

```txt
visa_mvtec_style/
├── candle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── anomaly/
│   └── ground_truth/
│       └── anomaly/
├── capsules/
├── cashew/
├── chewinggum/
├── fryum/
├── macaroni1/
├── macaroni2/
├── pcb1/
├── pcb2/
├── pcb3/
├── pcb4/
└── pipe_fryum/
```

Then point:

```python
DATA_ROOT = "/absolute/path/to/visa_mvtec_style"
```

#### Practical conversion rule

A simple rule is:
- use VisA normal training images as `train/good/`
- put normal test images into `test/good/`
- put anomalous test images into `test/anomaly/`
- put anomalous masks into `ground_truth/anomaly/`

The exact train/test split should follow the official VisA metadata CSV so that your evaluation matches the released benchmark. The `spot-diff` repository documents the official raw data tree, which is the safest source to follow when writing the conversion script. citeturn535417search5

#### VisA category list

The 12 VisA categories are:

```python
visa_categories = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]
```
  
---

## 5. Download the SAM checkpoint

For the best version of **SCL**, the script uses **SAM ViT-B**.

Download the **ViT-B SAM checkpoint** and save it as:

```txt
sam_vit_b_01ec64.pth
```

Place it somewhere accessible, for example:

```txt
project/
├── ucad_experiment.py
└── checkpoints/
    └── sam_vit_b_01ec64.pth
```

Then set:

```python
SAM_CHECKPOINT = "/absolute/path/to/checkpoints/sam_vit_b_01ec64.pth"
```

If you do not want to use SAM, set:

```python
SAM_CHECKPOINT = None
```

The code will then use **SLIC** instead of SAM.

---

## 6. Configure paths in the script

At the bottom of the script, update these two lines:

```python
DATA_ROOT = "/absolute/path/to/mvtec2d"
SAM_CHECKPOINT = "/absolute/path/to/sam_vit_b_01ec64.pth"
```

---

## 7. Run the experiment

### Default run

```bash
python visa.py
```

This will:
1. initialize the ViT prompt extractor
2. build the SAM-based or SLIC-based segmenter
3. train task by task over the 15 MVTec categories
4. build per-task memory after each task
5. evaluate on all seen tasks after each session
6. print metrics such as:
   - Image AUROC
   - Pixel AUROC
   - Pixel AUPR
   - forgetting measures

---


## 8. Expected console output

During training you should see logs similar to:

```txt
[device] Using: cuda
[init] Building extractor + segmenter…

=== Task 1/15: bottle ===
[inner] mean loss: ...
[outer] loss_q=... | align=... | scl=... | steps=...
[memory] bottle: keys=4096 | knowledge=20000
[calib] bottle: mode=zscore mu=... sigma=... thr=...
[eval] router=imgscore_ucad(...)
Testing (UCAD-style): 100%|██████████| ...

--- Results after task 'bottle' (1 seen) ---
Image AUROC: ...
Pixel AUROC: ...
Pixel AUPR (all pixels): ...
```

---

## 9. What each important argument means

### Data / model

- `categories`: order of continual tasks
- `data_root`: root folder of MVTec AD
- `image_size`: resize and crop size, default `224`
- `batch_size`: batch size for train/test loaders
- `sam_checkpoint`: path to SAM ViT-B weights, or `None`

### Memory

- `max_keys`: max number of early-layer key vectors per task
- `max_knowledge`: max number of final-layer knowledge vectors per task

### Meta-learning

- `inner_steps`: number of fast prompt update steps
- `outer_steps`: number of slow prompt update steps
- `lr_fast`: learning rate for fast prompts
- `lr_slow`: learning rate for slow prompts
- `align_coef`: weight of memory alignment loss
- `scl_coef`: weight of structure-aware contrastive loss

### Inference / scoring

- `pooling_mode`: `'count'` or `'ratio'`
- `image_topk`: top-k count or top-k ratio depending on mode
- `knn_k`: number of nearest neighbors used for patch scoring
- `use_calibration`: `None`, `'zscore'`, or `'robust'`
- `use_multiscale`: whether to fuse multiple scales
- `metric`: `'euclidean'` or `'cosine'`
- `center_quantile`: optional baseline subtraction before pooling

## Extra - Common issues and fixes

### Issue 1: `Category not found`

Error:

```txt
AssertionError: Category not found: ...
```

Fix:
- make sure `DATA_ROOT` points to the folder containing all 15 category folders
- make sure folder names match exactly, e.g. `metal_nut`, `toothbrush`

### Issue 2: SAM import error

Error:

```txt
ModuleNotFoundError: No module named 'segment_anything'
```

Fix:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or set:

```python
SAM_CHECKPOINT = None
```

### Issue 3: SAM checkpoint not found

Fix:
- verify the path to `sam_vit_b_01ec64.pth`
- make sure the file name is correct
- otherwise disable SAM and use SLIC fallback

### Issue 4: Out-of-memory on GPU

Try reducing:

```python
batch_size = 2
max_keys = 1024
max_knowledge = 5000
outer_steps = 2
use_multiscale = False
```

### Issue 5: very slow runtime

This is expected if:
- SAM is enabled
- many categories are used
- outer steps are large
- multiscale inference is enabled

For faster experiments:

```python
SAM_CHECKPOINT = None
use_multiscale = False
outer_steps = 2
batch_size = 4
```

---

## 15. Reproducibility

The script already fixes random seeds via:

```python
set_seed(1337)
```

This helps, but exact reproducibility can still vary depending on:
- GPU vs CPU
- CUDA / cuDNN versions
- PyTorch version
- whether SAM or SLIC is used

---

## Final checklist

Before running, confirm all of the following:

- [ ] `ucad_experiment.py` is saved
- [ ] Python environment is activated
- [ ] required packages are installed
- [ ] MVTec AD is downloaded and extracted
- [ ] `DATA_ROOT` points to the correct folder
- [ ] SAM checkpoint is downloaded, or `SAM_CHECKPOINT=None`
- [ ] category names in the script match the dataset folders
- [ ] you tested on 2–3 categories first
