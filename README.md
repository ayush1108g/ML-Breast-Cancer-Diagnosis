# Breast Cancer Detection ML Hackathon

This repository contains code for an ML hackathon focused on breast cancer detection. The goal of this hackathon is to develop an accurate model that can predict whether a breast mass is benign or malignant based on various features.

## Steps

1. Data Exploration
2. Model Selection
3. Model Preparation
4. Model Training

## Dataset

The dataset used for this hackathon contains 569 samples of breast mass features. Each sample has 30 features, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The dataset is available in the `data` directory in CSV format.

## Model Selection

For this challenge, I have chosen the Logistic Regression model. Logistic Regression is well-suited for binary classification tasks like this one. It's known for its simplicity and interpretability, making it an excellent choice for medical diagnosis.

## Model Preparation

Model preparation is crucial to ensure the machine learning model is ready for training.

### Data Normalization

To ensure consistent scales, the feature data is normalized using the Z-score normalization method. This step aids in efficient model training and convergence.

```python
def zscore_normalize(X):
    mu = np.mean(X, axis=0)
    sig = np.std(X, axis=0)
    X_norm = (X - mu) / sig
    return X_norm

X_train_array = zscore_normalize(X)
```
