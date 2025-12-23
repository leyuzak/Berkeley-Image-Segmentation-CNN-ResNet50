# Image Segmentation using Convolutional Neural Networks  
### Berkeley Segmentation Dataset (BSDS500)

This repository contains an end-to-end deep learning project for **image segmentation** using the **Berkeley Segmentation Dataset (BSDS500)**.  
The project compares a **custom-built Convolutional Neural Network (CNN)** with an **advanced transfer learning–based model using ResNet50**.

Both models are trained, evaluated, and deployed as **Streamlit applications** for interactive inference.

---

## 🎯 Project Objective

The main objectives of this project are:

- To build a **baseline CNN** for pixel-level boundary segmentation **without transfer learning**
- To develop an **advanced segmentation model using transfer learning with ResNet50**
- To preprocess image–mask pairs and convert raw annotations into binary segmentation masks
- To compare the performance of the baseline CNN and the transfer learning model
- To deploy trained models using **Streamlit** for real-time visualization and inference

This project focuses on understanding the **end-to-end image segmentation pipeline**, from data preprocessing to model evaluation and deployment.

---

## 📊 Dataset

- **Dataset:** Berkeley Segmentation Dataset (BSDS500)
- **Task:** Binary boundary segmentation
- **Annotations:** Human-labeled boundary masks
- **Preprocessing:**
  - Image resizing
  - Normalization
  - Binary mask generation
  - Train/validation/test split

---

## 🧠 Models Implemented

### 1️⃣ Baseline CNN (Custom Model)

- Built **from scratch** using TensorFlow / Keras
- No transfer learning
- Fully convolutional architecture
- Regularization with Dropout
- Loss Function: **Binary Cross-Entropy + Dice Loss**
- Optimized for boundary detection

📁 Folder:
berkeley/cnn_app/
---

### 2️⃣ Transfer Learning Model (ResNet50)

- Encoder based on **ResNet50 pretrained on ImageNet**
- Custom decoder layers for segmentation
- Fine-tuning applied to upper layers
- Loss Function: **Binary Cross-Entropy**
- Strong feature extraction with pretrained weights

📁 Folder:
berkeley/resnet_app/

---

## 📈 Results

### Quantitative Results (Test Set)

| Model | Input Size | Loss Function | Best Threshold | Validation IoU | Test IoU (Approx.) | Notes |
|------|-----------|--------------|---------------|----------------|-------------------|-------|
| Baseline CNN | 168 × 168 | BCE + Dice | 0.10 | ≈ 0.50 | ≈ 0.49–0.50 | Produces sharper boundaries |
| ResNet50 (TL) | 224 × 224 | BCE | 0.15–0.20 | ≈ 0.48 | ≈ 0.47–0.48 | Strong encoder, limited decoder |

### Qualitative Results
- Baseline CNN produces **sharper boundary predictions**
- ResNet50 captures **global structures** better but may lose fine details due to decoder limitations

---

## 🚀 Deployment

Both models are deployed using **Streamlit** for interactive segmentation demos.

Each app includes:
- `app.py` – Streamlit interface
- Saved model file (`.keras`)
- `requirements.txt`

The applications allow users to:
- Upload an image
- View predicted segmentation masks
- Adjust thresholds for visualization

---

## 📁 Project Structure

berkeley/
│
├── cnn_app/
│ ├── app.py
│ ├── cnn_model.keras
│ └── requirements.txt
│
├── resnet_app/
│ ├── app.py
│ ├── resnet50_model.keras
│ └── requirements.txt
│
└── image-segmentation-using-the-berkeley-segmentation.ipynb

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit
- Jupyter Notebook

---

## 📌 Notes

- This project was developed for an academic course on **Convolutional Neural Networks**
- No generative models were used
- The baseline model strictly avoids transfer learning
- The transfer learning model uses ResNet50 as required

