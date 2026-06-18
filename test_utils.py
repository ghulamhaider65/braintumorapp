# test_utils.py  — delete after testing
import numpy as np
from PIL import Image
from utils.inference import load_model, predict_tta
from utils.preprocess import preprocess_image, preprocess_mask
import io

# 1. Load model
print("Loading model...")
model = load_model("model/best_baseline_model.keras")
print("✅ Model loaded")

# 2. Create a dummy RGB image (simulates an upload)
dummy_img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
buf = io.BytesIO()
dummy_img.save(buf, format="PNG")
buf.seek(0)

# 3. Preprocess
img_batch = preprocess_image(buf)
print("✅ Preprocessed shape:", img_batch.shape)   # (1, 256, 256, 3)

# 4. TTA Predict
pred = predict_tta(model, img_batch)
print("✅ Prediction shape:", pred.shape)           # (256, 256, 1)
print("   Prediction range:", pred.min(), "–", pred.max())

# 5. Postprocess
soft_mask, binary_mask = preprocess_mask(pred[np.newaxis])
print("✅ Binary mask shape:", binary_mask.shape)   # (256, 256)
print("   Unique values:", np.unique(binary_mask))  # [0, 1]

print("\n✅ All utils working correctly — ready for Step 4 (app.py)")