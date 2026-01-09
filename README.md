# Image Data Exploration and Classification Using Transfer Learning

## Overview
This project explores image data and applies deep learning techniques for image classification as part of my MSc training in Biomedical Engineering. The objective is to gain hands-on experience in image preprocessing, dataset exploration, model development, and evaluation using modern convolutional neural networks.

Transfer learning with a pretrained ResNet50 model is used to study how high-level image features can be adapted to a new classification task.

---

## Objectives
- Explore and visualize image datasets and class distributions  
- Apply image preprocessing and data augmentation techniques  
- Train an image classification model using transfer learning  
- Evaluate model performance using clinically relevant metrics  

---

## Dataset Structure

dataset/
├── Training/
│ ├── class_1/
│ ├── class_2/
│ └── ...
└── Testing/
├── class_1/
├── class_2/
└── ...

---

## Methodology

### 1. Image Data Exploration
- Visualization of sample images  
- Analysis of class distributions  
- Inspection of dataset balance  

### 2. Image Preprocessing
- Image resizing to 128 × 128  
- Data normalization  
- Data augmentation (rotation, flipping, zooming)  
- Train/validation split  

### 3. Modeling
- Convolutional Neural Network using **ResNet50**  
- Pretrained ImageNet weights  
- Transfer learning with fine-tuning of selected layers  

### 4. Evaluation
Model performance is assessed using:
- Accuracy  
- Confusion matrix  
- Sensitivity (Recall)  
- Specificity  
- F1-score  

---

## Results
- The trained model successfully learned discriminative image features  
- Transfer learning improved convergence speed and classification performance  
- Evaluation metrics highlight strengths and limitations across classes  

---

## Learning Outcomes
- Practical experience in image preprocessing and augmentation  
- Understanding of transfer learning for image classification  
- Model evaluation beyond accuracy using sensitivity and specificity  
- Insight into strengths and limitations of deep learning models  

---

## Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib  
- scikit-learn  
- PIL  

---

## Notes
This project is intended for academic and learning purposes. Model performance depends on dataset quality and size, and results are not intended for deployment without further validation.
