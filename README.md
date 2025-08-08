## FSCIL_TCDS
Using requirements.txt for venv

## UCAD-main
Using environment.yaml for venv
For sam checkpoint: ```curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth```
For macOS: ```export KMP_DUPLICATE_LIB_OK=TRUE```
Use ```pip install faiss-cpu``` when running on CPU
On ```dataset_sam.py``` line 49 change device type to CPU if torch.cuda is not available
```conda install -c conda-forge opencv libtiff```

## Model files
FSCIL/UCAD-main/sam_vit_b_01ec64.pth - sam checkpoint
**fs_clip** - original version of fscil CLIP (from Suyu)
**oneclass_capsule** - init version (changes from fs_clip with SAM, metric ACC, train only on GOOD)
**original** - current match version (fast slow meta training with SAM, metric ACC + AUROC, train on BAD images as well)
**test_capsule** - variant of **orginal** (in trial progress)
**run_patchcore** - replicate UCAD (with memory bank, but current has OOM issue and take long time to run)
