# Task-2: Image Data Exploration & Classification

This project performs **image data exploration and image classification** using deep learning in Python.
It includes dataset loading, visualization, model training, and evaluation using standard performance metrics.

## Overview
- Load and explore training and testing image datasets
- Visualize class distributions
- Prepare data generators with augmentation
- Train an image classification model
- Apply **Transfer Learning using ResNet50 (ImageNet weights)**
- Evaluate the model using multiple metrics

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- PIL (Image Processing)
- Scikit-learn

## Dataset Structure
```
dataset/
 ├── Training/
 │    ├── class_1/
 │    ├── class_2/
 │    └── ...
 └── Testing/
      ├── class_1/
      ├── class_2/
      └── ...

      README.md
      txt
```

## Key Steps
1. **Data Exploration**
   - Count images per class
   - Visualize class distribution

2. **Data Preprocessing**
   - Image resizing (128x128)
   - Data augmentation
   - Train / validation split

3. **Modeling**
   - CNN and Transfer Learning using **ResNet50**
   - Pretrained ImageNet weights
   - Fine-tuning for classification

4. **Evaluation Metrics**
   - Accuracy
   - Confusion Matrix
   - Sensitivity (Recall)
   - Specificity
   - F1-Score

## How to Run
1. Install required libraries:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
   ```
2. Place the dataset in the correct folder structure.
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Execute all cells sequentially.

## Output
- Trained classification model
- Performance metrics and plots
- Saved model file

## Notes
This notebook is intended for academic and learning purposes.
