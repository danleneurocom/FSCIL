## FSCIL_TCDS
Using requirements.txt for venv

## UCAD-main
Using environment.yaml for venv
For sam checkpoint: ```curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth```
For macOS: ```export KMP_DUPLICATE_LIB_OK=TRUE```
Use ```pip install faiss-cpu``` when running on CPU
On ```dataset_sam.py``` line 49 change device type to CPU if torch.cuda is not available
```conda install -c conda-forge opencv libtiff```