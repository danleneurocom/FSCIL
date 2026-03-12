# UCAD-Style Continual Anomaly Detection

This repository contains code for **Continual anomaly detection** with **fast and slow prompt tuning**

The instructions below are aligned to the current root-level script:

```bash
python visa.py
```

---

## 1. Overview

The current pipeline implements:

* a **frozen ViT-B/16** backbone
* **fast / slow additive prompts** injected into each transformer block
* **inner / outer meta-training**
* **per-task memory**

  * `keys` from an early layer
  * `knowledge` from the final layer
* **task-agnostic inference** by selecting the task with the **lowest image anomaly score**
* **structure-aware contrastive learning (SCL)** using **SAM** with a **SLIC fallback**
* **continual evaluation** after each task on all seen categories

The script supports both:

* **VisA**
* **MVTec AD**

### Default setup in `visa.py`

The current `__main__` block is configured to run on **VisA** by default using these 12 categories:

```python
[
    'candle', 'capsules', 'cashew', 'chewinggum',
    'fryum', 'macaroni1', 'macaroni2',
    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]
```

If you want to run on **MVTec AD**, change:

```python
dataset = "mvtec"
```

and replace the category list with the 15 MVTec AD categories.

---

## 2. Recommended environment

Recommended setup:

* Python **3.10** or **3.11**
* PyTorch with CUDA if available
* GPU strongly recommended

The script automatically uses:

* `cuda` if available
* otherwise `mps` on Apple Silicon
* otherwise `cpu`

---

## 3. Create the environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

python -m pip install --upgrade pip
```

### Install dependencies

You can install from the repo requirements file:

```bash
pip install -r requirements.txt
```

Or install the main packages manually:

```bash
pip install torch torchvision numpy scikit-image scikit-learn tqdm Pillow
pip install opencv-python timm
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Notes

* `segment-anything` is optional. If it is not installed, the script falls back to **SLIC** for structure segmentation.
* `timm` is optional for the current default run.
* `opencv-python` is optional for most of the current execution path, but safe to install.

---

## 4. Prepare the dataset

This script supports both **VisA** and **MVTec AD**.

You only need to prepare the dataset you want to use.

---

## 5. Expected dataset structure

### 5.1 VisA

The current default run in `visa.py` uses:

```python
dataset = "visa"
```

The `VisA` loader in this repository expects the following structure:

```txt
DATA_ROOT/
├── candle/
│   └── Data/
│       └── Images/
│           ├── Normal/
│           ├── Anomaly/
│           └── Mask/
├── capsules/
│   └── Data/
│       └── Images/
│           ├── Normal/
│           ├── Anomaly/
│           └── Mask/
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

In other words, the script expects:

```python
root/category/Data/Images/Normal/*
root/category/Data/Images/Anomaly/*
root/category/Data/Images/Mask/*
```

If masks are missing, the script can still run, but missing masks will be treated as zero masks during pixel-level evaluation.

### 5.2 MVTec AD

If you switch to:

```python
dataset = "mvtec"
```

the `MVTecAD` loader expects:

```txt
DATA_ROOT/
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

This matches:

```python
root/category/train/good/*
root/category/test/<defect_type>/*
root/category/ground_truth/<defect_type>/*
```

Typical MVTec AD category list:

```python
[
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]
```

---

## 6. Download the SAM checkpoint

For the best version of **SCL**, the script can use **SAM ViT-B**.

Download the checkpoint:

```bash
curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

You can place it anywhere convenient, for example:

```txt
FSCIL/
├── visa.py
├── requirements.txt
└── sam_vit_b_01ec64.pth
```

Then set:

```python
SAM_CHECKPOINT = "/absolute/path/to/sam_vit_b_01ec64.pth"
```

If you do not want to use SAM, set:

```python
SAM_CHECKPOINT = None
```

The script will then use **SLIC** instead of SAM.

---

## 7. Configure paths in `visa.py`

At the bottom of `visa.py`, update the dataset and checkpoint paths in the `__main__` block.

### Example for VisA

```python
if __name__ == "__main__":
    DATA_ROOT = "/absolute/path/to/VisA_20220922"
    SAM_CHECKPOINT = "/absolute/path/to/sam_vit_b_01ec64.pth"
```

### Example without SAM

```python
if __name__ == "__main__":
    DATA_ROOT = "/absolute/path/to/VisA_20220922"
    SAM_CHECKPOINT = None
```

### Example for MVTec AD

```python
DATA_ROOT = "/absolute/path/to/mvtec2d"
```

and also change:

```python
dataset = "mvtec"
```

---

## 8. Run the experiment

### Default run

```bash
python visa.py
```

This will:

1. initialize the ViT prompt extractor
2. build the SAM-based or SLIC-based segmenter
3. train task by task over the selected categories
4. build per-task memory after each task
5. evaluate on all seen tasks after each session
6. print metrics such as:

   * Image AUROC
   * Pixel AUROC
   * Pixel AUPR
   * forgetting measures

---

## 9. Current default run settings

The current `visa.py` default configuration uses:

```python
run_ucad(
    categories=all_categories,
    data_root=DATA_ROOT,
    image_size=224,
    batch_size=8,
    sam_checkpoint=SAM_CHECKPOINT,
    max_keys=4096,
    max_knowledge=20000,
    inner_steps=5,
    outer_steps=5,
    lr_fast=3e-2,
    lr_slow=1e-4,
    align_coef=0.5,
    scl_coef=0.1,
    pooling_mode='ratio',
    image_topk=0.02,
    knn_k=3,
    use_calibration='zscore',
    use_multiscale=True,
    metric='euclidean',
    dataset='visa',
)
```

So by default, it runs with:

* **VisA**
* **224 × 224** input size
* **batch size 8**
* **top 2% patch pooling**
* **3-NN patch scoring**
* **z-score calibration**
* **multiscale inference enabled**

---

## 10. Expected console output

During training, you should see logs similar to:

```txt
[device] Using: cuda
[init] Building extractor + segmenter…

=== Task 1/12: candle ===
[inner] mean loss: ...
[outer] loss_q=... | align=... | scl=... | steps=...
[memory] candle: keys=4096 | knowledge=20000
[calib] candle: mode=zscore mu=... sigma=... thr=...
[eval] router=imgscore_ucad(...)
Testing (UCAD-style): 100%|██████████| ...

--- Results after task 'candle' (1 seen) ---
Image AUROC: ...
Pixel AUROC: ...
Pixel AUPR (all pixels): ...
```

If you switch to MVTec AD, the category names and number of tasks will change accordingly.

---

## 11. Important arguments

### Data / model

* `categories`: order of continual tasks
* `data_root`: root folder of the dataset
* `image_size`: resize and crop size, default `224`
* `batch_size`: batch size for train/test loaders
* `sam_checkpoint`: path to SAM ViT-B weights, or `None`
* `dataset`: either `'visa'` or `'mvtec'`

### Memory

* `max_keys`: maximum number of early-layer key vectors per task
* `max_knowledge`: maximum number of final-layer knowledge vectors per task

### Meta-learning

* `inner_steps`: number of fast prompt update steps
* `outer_steps`: number of slow prompt update steps
* `lr_fast`: learning rate for fast prompts
* `lr_slow`: learning rate for slow prompts
* `align_coef`: weight of memory alignment loss
* `scl_coef`: weight of structure-aware contrastive loss

### Inference / scoring

* `pooling_mode`: `'count'` or `'ratio'`
* `image_topk`: top-k count or top-k ratio depending on `pooling_mode`
* `knn_k`: number of nearest neighbors used for patch scoring
* `use_calibration`: `None`, `'zscore'`, or `'robust'`
* `use_multiscale`: whether to fuse multiple scales
* `metric`: `'euclidean'` or `'cosine'`
* `center_quantile`: optional baseline subtraction before pooling

---

## 12. Common issues and fixes

### Issue 1: VisA path error

Error example:

```txt
AssertionError: [VisA] Missing: ...
```

Fix:

* make sure `DATA_ROOT` points to the folder containing all 12 VisA category folders
* make sure each category contains:

  * `Data/Images/Normal`
  * `Data/Images/Anomaly`
* if masks are available, place them under:

  * `Data/Images/Mask`

### Issue 2: MVTec category path error

Error example:

```txt
AssertionError: Category not found: ...
```

Fix:

* make sure `DATA_ROOT` points to the folder containing all MVTec category folders
* make sure folder names match exactly, for example:

  * `metal_nut`
  * `toothbrush`

### Issue 3: SAM import error

Error:

```txt
ModuleNotFoundError: No module named 'segment_anything'
```

Fix:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or disable SAM:

```python
SAM_CHECKPOINT = None
```

### Issue 4: SAM checkpoint not found

Fix:

* verify the path to `sam_vit_b_01ec64.pth`
* make sure the file name is correct
* otherwise disable SAM and use the SLIC fallback

### Issue 5: torchvision too old

Error example:

```txt
Please install/upgrade torchvision>=0.13 to use ViT_B_16.
```

Fix:

* upgrade `torchvision`
* reinstall `torch` and `torchvision` with matching versions

### Issue 6: Out-of-memory on GPU

Try reducing:

```python
batch_size = 2
max_keys = 1024
max_knowledge = 5000
outer_steps = 2
use_multiscale = False
```

You can also test on fewer categories first.

### Issue 7: Very slow runtime

This is expected when:

* SAM is enabled
* many categories are used
* `outer_steps` is large
* multiscale inference is enabled

For faster experiments:

```python
SAM_CHECKPOINT = None
use_multiscale = False
outer_steps = 2
batch_size = 4
```

---

## 13. Reproducibility

The script sets random seeds via:

```python
set_seed(1337)
```

This improves reproducibility, although exact results can still vary depending on:

* GPU vs CPU
* CUDA / cuDNN versions
* PyTorch version
* whether SAM or SLIC is used

---

## 14. Repository structure

A simplified view of the current repository root:

```txt
FSCIL/
├── README.md
├── requirements.txt
├── visa.py
├── main.py
├── app.py
├── loadMVTec.py
├── patchcore.py
├── patchcoresam.py
├── patchcoresamfs.py
├── FS_SAM/
├── UCAD-main/
└── ...
```

The instructions in this README are specifically intended for the current root-level `visa.py` script.

---

## 15. Final checklist

Before running, confirm all of the following:

* [ ] `visa.py` is available in the repo root
* [ ] the Python environment is activated
* [ ] required packages are installed
* [ ] the dataset is downloaded and extracted
* [ ] `DATA_ROOT` points to the correct folder
* [ ] `dataset` is set correctly to `'visa'` or `'mvtec'`
* [ ] the SAM checkpoint is downloaded, or `SAM_CHECKPOINT = None`
* [ ] category names in the script match the dataset folders
* [ ] you tested on 1 to 3 categories first before a full run
