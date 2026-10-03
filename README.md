# 🫁 TB Chest Radiography — Tuberculosis Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images as **Normal** or **Tuberculosis** using a fine-tuned **ResNet50** convolutional neural network, with **Grad-CAM** visualizations to highlight the lung regions driving each prediction. Includes an interactive web demo built with **Gradio**.

![Tech](https://img.shields.io/badge/Tech-TensorFlow%20%7C%20Keras-orange)
![Model](https://img.shields.io/badge/Model-ResNet50-blue)
![Status](https://img.shields.io/badge/Status-Research%2FLearning%20Project-lightgrey)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Disclaimer](#-disclaimer)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🚀 Overview

Tuberculosis (TB) remains a significant global health concern, and chest X-rays are one of the most common tools used for screening. This project explores using a convolutional neural network to automatically classify chest X-ray images as **Normal** or **Tuberculosis**, with **explainability** built in via Grad-CAM — so predictions aren't just a black-box label, but come with a visual heatmap showing *which parts of the lung* most influenced the model's decision.

---

## 🔍 How It Works

1. A chest X-ray image is uploaded
2. The image is converted to grayscale, resized to 256×256, and normalized
3. The preprocessed image is passed through a trained **ResNet50** classifier
4. The model outputs a probability of Tuberculosis vs Normal
5. **Grad-CAM** (Gradient-weighted Class Activation Mapping) is computed from the last convolutional layer of ResNet50, producing a heatmap of the regions most responsible for the prediction
6. The heatmap is overlaid on the original X-ray so the result is visually interpretable, not just a raw score

---

## 🗂️ Dataset

The project uses chest X-ray images with accompanying metadata:
- `Normal.metadata.xlsx` — metadata for normal chest X-ray samples
- `Tuberculosis.metadata.xlsx` — metadata for tuberculosis-positive chest X-ray samples

*(This matches the structure of the widely-used public Tuberculosis Chest X-ray dataset — update this section with the exact source/citation if your dataset came from a specific public repository, e.g. Kaggle, so proper credit is given.)*

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Model architecture | ResNet50 (transfer learning / fine-tuned CNN) |
| Deep learning framework | TensorFlow / Keras |
| Image processing | OpenCV (`cv2`) |
| Numerical computing | NumPy |
| Explainability | Grad-CAM |
| Web demo interface | Gradio |
| Training/experimentation | Jupyter Notebook |

---

## 📂 Project Structure

```
TB_Chest_Radiography/
│
├── tuberulosis.ipynb              # Model training & experimentation notebook
├── tb_detection_resnet50.keras    # Trained ResNet50 model (generated after training)
├── app_gradio.py                  # Gradio web demo with Grad-CAM visualization
├── app_t.py                       # Additional/alternate app script
├── predict.py                     # CLI script for single-image prediction
├── Normal.metadata.xlsx           # Metadata for normal X-ray samples
├── Tuberculosis.metadata.xlsx     # Metadata for TB-positive X-ray samples
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/ShivangiSingh13/TB_Chest_Radiography.git
cd TB_Chest_Radiography
pip install tensorflow opencv-python numpy gradio
```

> If you have a `requirements.txt`, use `pip install -r requirements.txt` instead — add one to the repo if it doesn't exist yet, so setup is reproducible for anyone cloning it.

### Model File

The Gradio app and prediction script both expect a trained model file named:
```
tb_detection_resnet50.keras
```
in the project root. If this file isn't included in the repo (model files are often excluded due to size), train the model by running through `tuberulosis.ipynb` first, or download it separately and place it in the root directory.

---

## ▶️ Usage

### Run the Gradio web demo

```bash
python app_gradio.py
```

This launches an interactive web interface where you can:
- Upload a chest X-ray image
- View the predicted class (Normal / Tuberculosis) with confidence scores
- See a Grad-CAM heatmap overlay highlighting the regions that influenced the prediction

### Run a single prediction from the command line

```bash
python predict.py path/to/xray_image.png
```

Example output:
```
path/to/xray_image.png -> Tuberculosis Detected 🟥 (prob=0.873)
```

---

## 🧠 Model Details

- **Architecture**: ResNet50 (likely with transfer learning from ImageNet weights, fine-tuned on the TB dataset)
- **Input size**: 256×256, converted to grayscale then replicated to 3 channels (to match ResNet50's expected input shape)
- **Output**: Binary classification (sigmoid output — probability of Tuberculosis)
- **Explainability**: Grad-CAM computed from the `conv5_block3_out` layer (ResNet50's final convolutional block)

For full training details — architecture configuration, data augmentation, training/validation split, and evaluation metrics (accuracy, precision, recall, AUC) — see `tuberulosis.ipynb`.

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. It is **not a certified diagnostic tool** and should not be used as a substitute for professional medical evaluation. Any real-world screening or diagnosis of Tuberculosis should be performed by qualified healthcare professionals using validated clinical tools.

---

## 📌 Future Improvements

- Add quantitative evaluation metrics (accuracy, precision, recall, F1, AUC-ROC) to this README once finalized from the notebook
- Add a `requirements.txt` for reproducible installs
- Experiment with additional architectures (EfficientNet, DenseNet) for comparison
- Add a proper train/validation/test split report and confusion matrix
- Deploy the Gradio demo publicly (e.g. Hugging Face Spaces) for easier access without local setup
- Add unit tests for the preprocessing and prediction pipeline

---

## 📄 License

This project is developed for learning and research purposes.
