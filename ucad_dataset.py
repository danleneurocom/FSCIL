import os
from PIL import Image
from torch.utils.data import Dataset

class UCADDataset(Dataset):
    # Loads images from mvtec2d-sam-b/<class>/(train|test)/<subclass>/*.png
    def __init__(self, root_dir, subdatasets, split="train", transform=None):

        # subdatasets: list of category names, e.g. ['bottle','cable',...]
        self.samples = []
        for idx, cat in enumerate(subdatasets):
            split_dir = os.path.join(root_dir, cat, split)
            # each split_dir may have subfolders like 'good', 'defectA', ...
            for subclass in os.listdir(split_dir):
                subclass_dir = os.path.join(split_dir, subclass)
                if not os.path.isdir(subclass_dir): continue
                for fname in os.listdir(subclass_dir):
                    if fname.lower().endswith(".png"):
                        path = os.path.join(subclass_dir, fname)
                        self.samples.append((path, idx))
        self.transform = transform
        self.class_names = subdatasets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label