import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASE_DIR = "/content/drive/MyDrive/major_project"
PATIENT_DIR = os.path.join(BASE_DIR, "patient_list")
OUT_DIR = os.path.join(BASE_DIR, "dataset_2d")

os.makedirs(OUT_DIR, exist_ok=True)

patients = sorted(os.listdir(PATIENT_DIR))

# -------------------------
# PATIENT-LEVEL SPLIT
# -------------------------
train_p, temp_p = train_test_split(patients, test_size=0.3, random_state=42)
val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)

splits = {
    "train": train_p,
    "val": val_p,
    "test": test_p
}

for split in splits:
    os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, split, "masks"), exist_ok=True)

# -------------------------
# 3D → 2D CONVERSION
# -------------------------
for split, plist in splits.items():
    print(f"\n🔄 Processing {split} patients")

    for patient in tqdm(plist):
        p_dir = os.path.join(PATIENT_DIR, patient)

        img_path = [f for f in os.listdir(p_dir) if "t1c" in f.lower()][0]
        seg_path = [f for f in os.listdir(p_dir) if "seg" in f.lower()][0]

        img = nib.load(os.path.join(p_dir, img_path)).get_fdata()
        seg = nib.load(os.path.join(p_dir, seg_path)).get_fdata()

        for i in range(img.shape[2]):
            img_slice = img[:, :, i]
            seg_slice = seg[:, :, i]

            if np.max(seg_slice) == 0:
                continue  # skip empty slices

            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            seg_slice = (seg_slice > 0).astype(np.float32)

            np.save(f"{OUT_DIR}/{split}/images/{patient}_slice_{i}.npy", img_slice)
            np.save(f"{OUT_DIR}/{split}/masks/{patient}_slice_{i}.npy", seg_slice)

print("\n✅ 3D → 2D conversion complete")
