# Diabetic Retinopathy Detection 

A deep learning solution using Vision Transformers to classify stages of diabetic retinopathy from retinal fundus images.

---

## Project Overview

Diabetic retinopathy is a leading cause of vision impairment. Early detection is essential for effective treatment. This project implements a Vision Transformer (ViT) model to classify retinal images into five DR severity levels, aiding medical professionals in diagnosing and staging the disease.

---

## Tech Stack

- **Programming & Data Tools:** Python, OpenCV, NumPy, Pandas  
- **Deep Learning:** TensorFlow, Keras  
- **Model Architecture:** Vision Transformer (ViT)  
- **Environment:** Jupyter Notebook

---

## Dataset

The dataset consists of labeled retinal fundus images categorized into five DR severity classes (0–4). Image preprocessing includes:

- Resizing images to ViT input dimensions  
- Histogram equalization and normalization  
- Data augmentation: rotations, flips, zooms, brightness changes
- [Dataset link](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)

---

## Methodology

1. **Data Preprocessing**  
   - Load and inspect images  
   - Apply resizing, normalization, and augmentation for robust training

2. **ViT Model Setup**  
   - Construct Vision Transformer architecture with patch embedding  
   - Use transformer encoder blocks + classification head

3. **Training & Evaluation**  
   - Train on preprocessed images  
   - Use callbacks like early stopping and learning rate scheduling  
   - Evaluate using accuracy, precision, recall, and confusion matrix plots

4. **Model Tuning**  
   - Fine-tune hyperparameters (learning rate, epochs, batch size)  
   - Apply dropout and regularization to avoid overfitting

---

##  Results & Insights

- Achieved high classification accuracy (actual metrics here, e.g., **<X.%>**)  
- Model reliably distinguished between DR stages  
- Confusion matrix highlighted challenges in borderline cases (e.g., stage 2 vs 3)  
- Visualization techniques (e.g., Grad-CAM) helped interpret model focus areas

---

##  Quick Start

```bash
git clone https://github.com/Siddhart2004/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

jupyter notebook ViT_DR_Detection.ipynb

