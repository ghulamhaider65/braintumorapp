# Brain Tumor Segmentation

A Streamlit app for brain MRI tumor segmentation using a pre-trained Attention U-Net model. The app lets you upload an MRI scan, run inference, visualize the predicted tumor mask, and download the results.

## Features

- Upload MRI images in JPG, PNG, BMP, or TIFF format
- Run segmentation with a saved TensorFlow/Keras model
- Optional test-time augmentation (TTA) for more stable predictions
- Adjustable detection threshold
- Visual outputs:
  - Original MRI
  - Soft prediction map
  - Binary mask
  - Tumor overlay
- Download the binary mask and overlay image

## Model

The app loads the pre-trained model from:

- [model/best_baseline_model.keras](model/best_baseline_model.keras)

Training and experimentation were done in the Kaggle notebook below:

- [BrainTumor Segmentation Baseline vs AttentionU-Net](https://www.kaggle.com/code/ghulamhiader/braintumor-segmentation-baseline-vs-attentionu-net/notebook)

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── test_utils.py
├── model/
│   └── best_baseline_model.keras
└── utils/
    ├── __init__.py
    ├── inference.py
    └── preprocess.py
```

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## How It Works

- `utils/preprocess.py` handles image and mask preprocessing.
- `utils/inference.py` loads the model, runs prediction, and computes summary statistics.
- `app.py` provides the Streamlit interface and visualization.

## Notes

- The model is intended for research and demonstration purposes only.
- This project is not a medical diagnostic tool.
- Best results are expected with brain MRI images similar to the training data used in the Kaggle notebook.
