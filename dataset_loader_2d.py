# dataset_loader_2d.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2   # 🔥 REQUIRED

class BrainTumorDataset(Dataset):
    def __init__(self, base_dir, split="train", target_size=160):
        """
        base_dir : path to dataset_2d
        split    : train / val / test
        """
        self.image_dir = os.path.join(base_dir, split, "images")
        self.mask_dir  = os.path.join(base_dir, split, "masks")
        self.target_size = target_size

        assert os.path.exists(self.image_dir), f"Missing {self.image_dir}"
        assert os.path.exists(self.mask_dir),  f"Missing {self.mask_dir}"

        self.images = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".npy")]
        )
        self.masks = sorted(
            [f for f in os.listdir(self.mask_dir) if f.endswith(".npy")]
        )

        assert len(self.images) > 0, "❌ No images found"
        assert len(self.images) == len(self.masks), "❌ Image-mask mismatch"

        print(f"✅ {split.upper()} loaded slices: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load
        img = np.load(os.path.join(self.image_dir, self.images[idx]))
        mask = np.load(os.path.join(self.mask_dir, self.masks[idx]))

        # 🔥 RESIZE (MAIN SPEED FIX)
        img = cv2.resize(
            img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Normalize image
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Add channel dim → (1, H, W)
        img = np.expand_dims(img, axis=0)

        # Binary mask
        mask = (mask > 0).astype(np.float32)

        return torch.tensor(img), torch.tensor(mask)
