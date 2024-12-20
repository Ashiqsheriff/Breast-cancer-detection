# Breast Cancer Cell Segmentation

## Overview
This project focuses on the semantic segmentation of breast cancer cell images using deep learning techniques. The goal is to accurately identify and delineate cancerous regions in microscopic images of breast tissue. The project compares segmentation performance between models trained on normal images and those enhanced with Canny edge overlays.

## Table of Contents
1. [Background](#background)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [Usage](#usage)
9. [Future Work](#future-work)
10. [Acknowledgments](#acknowledgments)

---

## Background
Breast cancer is one of the leading causes of cancer-related deaths globally. Accurate segmentation of cancer cells from microscopic images is critical for diagnosis, treatment planning, and prognosis. This project employs U-Net, a well-known architecture for biomedical image segmentation, to achieve precise segmentation results.

---

## Dataset
The dataset consists of:
- **Normal Images**: Original breast cancer cell images.
- **Canny Overlay Images**: Images enhanced with Canny edge detection to highlight boundaries.

### Directory Structure
- Training Images: `../content/drive/MyDrive/breast cancer cell/train folder/img/`
- Training Masks: `../content/drive/MyDrive/breast cancer cell/train folder/masks/`
- Validation Images: `../content/drive/MyDrive/breast cancer cell/validation folder/img/`
- Validation Masks: `../content/drive/MyDrive/breast cancer cell/validation folder/masks/`

Augmented images are stored in:
`/content/drive/MyDrive/breast cancer cell/train folder/img/augmented`

---

## Preprocessing
- **Normalization**: Input images are normalized to scale pixel values between 0 and 1.
- **Data Augmentation**: Performed to enhance model generalization due to the small dataset size.
- **Edge Detection**: Canny edge detection applied to create overlay images.

---

## Model Architecture
The project uses a U-Net architecture:
- **Input**: Single-channel grayscale images.
- **Loss Function**: Binary Crossentropy.
- **Optimizer**: Adam with a learning rate of 1e-4.
- **Metrics**: 
  - Mean Intersection over Union (IoU)
  - Accuracy

---

## Training
The model was trained using:
- **Normal Images**: Baseline segmentation.
- **Canny Overlay Images**: Enhanced segmentation using boundary features.

Optimizer Parameters:
- Learning rate: `1E-4`
- Beta1: `0.9`
- Beta2: `0.999`
- Epsilon: `1e-08`

---

## Results
- **Normal Images**: Achieved **80% segmentation accuracy**.
- **Canny Overlay Images**: Achieved **92% segmentation accuracy**.
- IoU values showed significant improvement with Canny overlays.

---

## Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- OpenCV
- Google Colab (for implementation)

---

## Usage
### Steps to Run:
1. Clone the repository or download the project files.
2. Set up the directory structure and upload the dataset to the specified paths.
3. Install required dependencies using `pip install -r requirements.txt`.
4. Train the model using the script:
   ```bash
   python train.py
