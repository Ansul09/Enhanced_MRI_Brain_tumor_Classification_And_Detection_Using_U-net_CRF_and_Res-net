import streamlit as st
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label

from unet_model import UNet
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.25

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation (Patient-Level)")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load("unet_trained (5).pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()
st.success("✅ Model loaded")

# -----------------------------
# HELPERS
# -----------------------------
def normalize(volume):
    return (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

def remove_small_components(mask, min_size=50):
    labeled, num = label(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, num + 1):
        if np.sum(labeled == i) >= min_size:
            cleaned[labeled == i] = 1
    return cleaned

# -----------------------------
# FILE UPLOAD (🔥 FIXED)
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload FULL PATIENT MRI (T1c .nii.gz)",
    type=["nii", "gz", "npy"]   # ✅ FIX
)

if uploaded_file is None:
    st.info("⬆ Upload a full patient MRI volume (T1c)")
    st.stop()

# -----------------------------
# LOAD VOLUME (3D)
# -----------------------------
if uploaded_file.name.endswith(".npy"):
    volume = np.load(uploaded_file)

elif uploaded_file.name.endswith(".nii") or uploaded_file.name.endswith(".nii.gz"):
    suffix = ".nii.gz" if uploaded_file.name.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    nii = nib.load(tmp_path)
    volume = nii.get_fdata()



else:
    st.error("❌ Unsupported file type")
    st.stop()

# Handle possible (H,W,D,1)
if volume.ndim == 4:
    volume = volume[..., 0]

if volume.ndim != 3:
    st.error(f"❌ Expected 3D volume, got shape {volume.shape}")
    st.stop()

# Convert to (D, H, W)
volume = normalize(volume)
volume = np.transpose(volume, (2, 0, 1))

valid_indices=[
    i for i in range(volume.shape[0])
    if volume[i].mean()>0.05
]

volume= volume[valid_indices]

st.success(f"Using {len(valid_indices)} informative slices out of full volume")
st.success(f"📦 Loaded patient volume: {volume.shape}")

# -----------------------------
# INFERENCE (WHOLE PATIENT)
# -----------------------------
pred_slices = []

with torch.no_grad():
    for i in range(volume.shape[0]):
        slice_img = volume[i]

        inp = torch.tensor(slice_img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        prob = torch.sigmoid(model(inp))
        pred = (prob > THRESHOLD).float()

        pred_slices.append(pred.cpu().numpy()[0, 0])

pred = np.stack(pred_slices, axis=0)


pred = remove_small_components(pred)

# -----------------------------
# SLICE VIEWER
# -----------------------------
slice_idx = st.slider(
    "Select slice",
    0,
    volume.shape[0] - 1,
    volume.shape[0] // 2
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("MRI Slice")
    fig, ax = plt.subplots()
    ax.imshow(volume[slice_idx], cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

with col2:
    st.subheader("Tumor Overlay")
    fig, ax = plt.subplots()
    ax.imshow(volume[slice_idx], cmap="gray")
    ax.imshow(pred[slice_idx], cmap="Reds", alpha=0.4)
    ax.axis("off")
    st.pyplot(fig)

# -----------------------------
# SAVE OUTPUT
# -----------------------------
if st.button("💾 Save Prediction"):
    os.makedirs("predictions", exist_ok=True)
    np.save("predictions/patient_prediction.npy", pred)
    st.success("✅ Saved predictions/patient_prediction.npy")

# -----------------------------
# STATS
# -----------------------------
st.markdown(f"### 🧪 Tumor voxels detected: **{int(pred.sum())}**")
