import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MVTecImageLevel(Dataset):
    def __init__(self, root_dir, split="train", category="capsule", transform=None):
        """
        root_dir/
          <category>/
            train/good/*.png
            test/{good,crack,poke,…}/*.png
            ground_truth/…      <-- ignored
        """
        self.samples = []
        self.transform = transform
        split_dir = os.path.join(root_dir, category, split)
        for sub in os.listdir(split_dir):
            subdir = os.path.join(split_dir, sub)
            if not os.path.isdir(subdir):
                continue
            label = 0 if sub == "good" else 1
            for fname in os.listdir(subdir):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(subdir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
