import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io
import tensorflow as tf

from utils.preprocess import preprocess_image, preprocess_mask
from utils.inference import load_model, predict_tta, compute_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model (cached — only runs once) ─────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model("model/best_baseline_model.keras")

model = get_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Brain Tumor Detector")
    st.markdown("---")
    st.markdown("### Model Info")
    st.info("**Architecture:** Attention U-Net\n\n"
            "**Input size:** 256 × 256\n\n"
            "**Val Dice:** 0.7980\n\n"
            "**Val IoU:** 0.6690")

    st.markdown("---")
    st.markdown("### Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        help="Lower = more sensitive (more detections). Higher = more conservative."
    )
    use_tta = st.checkbox(
        "Use TTA (Test-Time Augmentation)",
        value=True,
        help="Averages 4 flipped predictions for better accuracy. Slightly slower."
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a brain MRI scan\n"
        "2. Wait for the model to process\n"
        "3. View the segmentation result\n"
        "4. Adjust threshold if needed"
    )
    st.markdown("---")
    st.warning("⚠️ For research purposes only. Not a medical diagnostic tool.")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🧠 Brain Tumor Segmentation")
st.markdown("Upload a brain MRI image to detect and segment tumor regions.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, PNG, BMP, TIFF"
)

if uploaded_file is None:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Architecture", "Attention U-Net")
    with col2:
        st.metric("Val Dice Score", "0.7980")
    with col3:
        st.metric("Val IoU Score", "0.6690")

    st.info("👆 Upload an MRI scan above to get started.")

else:
    # ── Inference ─────────────────────────────────────────────────────────────
    original_img = Image.open(uploaded_file).convert("RGB")
    original_size = original_img.size  # (width, height)

    with st.spinner("🔍 Analyzing MRI scan..."):
        # Preprocess
        uploaded_file.seek(0)
        img_batch = preprocess_image(uploaded_file)

        # Predict
        if use_tta:
            pred_raw = predict_tta(model, img_batch)          # (256,256,1)
            pred_batch = pred_raw[np.newaxis]                  # (1,256,256,1)
        else:
            pred_batch = model.predict(img_batch, verbose=0)   # (1,256,256,1)

        # Postprocess
        soft_mask, binary_mask = preprocess_mask(pred_batch, threshold=threshold)

        # Stats
        stats = compute_stats(binary_mask, original_size)

    # ── Results banner ────────────────────────────────────────────────────────
    st.markdown("---")
    if stats["tumor_detected"]:
        st.error(f"🔴 **Tumor Detected** — {stats['tumor_percent']}% of scan area")
    else:
        st.success("🟢 **No Tumor Detected** — No significant tumor region found")

    # ── Metrics row ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tumor Detected", "Yes" if stats["tumor_detected"] else "No")
    with col2:
        st.metric("Area Coverage", f"{stats['tumor_percent']}%")
    with col3:
        st.metric("Tumor Pixels (256²)", stats["tumor_pixels_resized"])
    with col4:
        st.metric("Method", "TTA ×4" if use_tta else "Single Pass")

    # ── Visualization ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Segmentation Results")

    # Prepare overlay
    img_resized = np.array(original_img.resize((256, 256))).astype(np.float32) / 255.0
    overlay = img_resized.copy()
    overlay[binary_mask == 1, 0] = 1.0   # Red channel → tumor region
    overlay[binary_mask == 1, 1] *= 0.3
    overlay[binary_mask == 1, 2] *= 0.3

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor('#0E1117')

    panels = [
        (np.array(original_img.resize((256, 256))), "Original MRI",      None),
        (soft_mask,                                   "Soft Prediction",   'hot'),
        (binary_mask,                                 f"Binary Mask (t={threshold})", 'gray'),
        (overlay,                                     "Tumor Overlay",     None),
    ]

    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.axis('off')
        ax.set_facecolor('#0E1117')

    # Legend for overlay panel
    red_patch = mpatches.Patch(color='red', label='Tumor Region')
    axes[3].legend(handles=[red_patch], loc='lower right',
                   facecolor='#1E2130', labelcolor='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Download section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Download Results")

    col1, col2 = st.columns(2)

    with col1:
        # Save mask as PNG
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Binary Mask",
            data=buf.getvalue(),
            file_name="tumor_mask.png",
            mime="image/png"
        )

    with col2:
        # Save overlay as PNG
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        buf2 = io.BytesIO()
        overlay_img.save(buf2, format="PNG")
        st.download_button(
            "⬇️ Download Overlay Image",
            data=buf2.getvalue(),
            file_name="tumor_overlay.png",
            mime="image/png"
        )
