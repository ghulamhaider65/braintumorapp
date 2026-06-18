import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 256

def preprocess_image(uploaded_file):
    """
    Takes a streamlit uploaded file object, 
    returns a numpy array ready for model input.
    Mirrors exactly what was done during training.
    """
    # Read uploaded file PIL to numpy array
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img, dtype = np.uint8)

    # Resize to 256x256
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize to [0, 1]
    img = np.expand_dims(img, axis = 0)

    return img

def preprocess_mask(pred, threshold = 0.5):
    """
    Takes raw output (1, 256, 256, 1),
    returns binary mask (256, 256) as uint8.
    """
    mask = pred[0, :, :, 0]
    mask_binary = (mask > threshold).astype(np.uint8)
    return mask, mask_binary
