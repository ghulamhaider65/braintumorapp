import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# custom objects used during training

def dice_coef(y_true, y_pred):
    y_true = K.cast(K.flatten(y_true), "float32")
    y_pred = K.cast(K.flatten(y_pred), "float32")
    y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def iou(y_true, y_pred):
    y_true = K.cast(K.flatten(y_true), "float32")
    y_pred = K.cast(K.flatten(y_pred), "float32")
    y_pred = K.clip(y_pred, 1e-7, 1.0 - 1e-7)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + 1.) / (union + 1.)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def weighted_bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        K.cast(y_true, "float32"),
        K.cast(y_pred, "float32")
    )
    weight_map = K.cast(y_true, "float32") * 9.0 + 1.0
    weighted_bce = K.mean(bce * K.squeeze(weight_map, axis=-1))
    return weighted_bce + dice_loss(y_true, y_pred)

@tf.function(reduce_retracing = True)
def _predict_single(model, x):
    return model(x, training = False)

def load_model(model_path: str):
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects = {
            "dice_coef": dice_coef,
            "iou": iou,
            "weighted_bce_dice_loss": weighted_bce_dice_loss
        }
    )
    return model

def predict_tta(model, img_batch):
    """
    img_batch shape (1, 256, 256, 1)"""

    img = img_batch[0]

    augmented = [
        img, 
        img[:, ::, :],
        img[::-1, :, :],
        img[::-1, ::-1, :],
    ]
    preds = [
        model.predict(i[np.newaxis], verbose = 0)[0] for i in augmented
    ]

    preds[1] = preds[1][:, ::-1, :]
    preds[2] = preds[2][::-1, :, :]
    preds[3] = preds[3][::-1, ::-1, :]

    return np.mean(preds, axis = 0)

def compute_stats(mask_binary, original_size):
    """
    mask_binary : (256, 256)
    original_size : (height, width)
    """

    total_pixels = mask_binary.size
    total_pixels = mask_binary.sum()
    tumor_pixels = mask_binary.sum()
    tumor_percent = (tumor_pixels / total_pixels) * 100.0
    
    # Approximate real-world area assuming standart MRI slice thickness of 5mm and pixel spacing of 1mm

    orig_w, orig_h = original_size
    scale_x = orig_w / 256
    scale_y = orig_h / 256
    tumor_pixels_original = tumor_pixels * scale_x * scale_y

    return {
        "tumor_detected": tumor_pixels > 50,
        "tumor_percent": round(tumor_percent, 2),
        "tumor_pixels_resized": int(tumor_pixels),
        "tumor_pixels_original": int(tumor_pixels_original)
    }