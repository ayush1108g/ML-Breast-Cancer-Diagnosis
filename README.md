# Contributers

### Ayush Gupta 
### Vivek Singh
### Divya Kumar

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

### Sigmoid Function

The sigmoid function is defined to calculate the probability of a breast mass being malignant or benign.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Cost Function

The cost function is designed to compute the loss or error of the model. It includes both the loss and regularization term.

### Gradient Calculation

The gradient of the cost function is computed to guide model parameter updates during training. This gradient includes both the data-driven gradient and the regularization term gradient.


## Model Training

### Parameters

Initialize the model parameters, weights (w) and bias (b) with zeros or small random values.
```python
w_tmp = np.zeros_like(X_train_array[0])
b_tmp = 0.0
```

### HyperParameters

Define Hyperparameters such as the learning rate (alpha), the number of iterations (iters), and the regularization parameter (lambda_).
```python
alpha = 0.01
iters = 4000
lambda_ = 0.3
```
### Gradient Descent

Training of the model using gradient descent, where the cost is minimized iteratively by updating the parameters.

### Model Evaluation

```python
p = predict(X_train_array, w, b)
print('Train Accuracy: %f' % (np.mean(p == y_train) * 100))
```
## train_accuracy : 98.594025


Now that we have obtained the final parameters w and b, you can utilize the "zscore_normalize(X)" function to preprocess your test data. We can employ the "predict" function to determine whether a given patient's cancer diagnosis is malignant or benign based on their complete set of feature values.


### M-L B-R-E-A-S-T--C-A-N-C-E-R--D-E-T-E-C-T-I-O-N
