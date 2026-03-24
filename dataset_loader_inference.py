# dataset_loader_inference.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainTumorInferenceDataset(Dataset):
    """
    Dataset for inference.
    Expected structure:
    base_dir/
    ├── images/
    │   ├── patient001_slice001.npy
    │   ├── patient001_slice002.npy
    └── masks/
        ├── patient001_slice001.npy
        ├── patient001_slice002.npy
    """

    def __init__(self, base_dir):
        self.image_dir = os.path.join(base_dir, "images")
        self.mask_dir = os.path.join(base_dir, "masks")

        assert os.path.exists(self.image_dir), f"❌ Missing: {self.image_dir}"
        assert os.path.exists(self.mask_dir), f"❌ Missing: {self.mask_dir}"

        self.images = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".npy"))
        self.masks  = sorted(f for f in os.listdir(self.mask_dir) if f.endswith(".npy"))

        assert len(self.images) == len(self.masks), "❌ Image–mask count mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_dir, self.images[idx])).astype(np.float32)
        mask  = np.load(os.path.join(self.mask_dir, self.masks[idx])).astype(np.float32)

        # Ensure shapes
        if image.ndim == 2:
            image = image[None, :, :]   # (1, H, W)
        if mask.ndim == 3:
            mask = mask.squeeze()

        # 🔑 Extract patient ID (before first underscore)
        patient_id = self.images[idx].split("_")[0]

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            patient_id
        )