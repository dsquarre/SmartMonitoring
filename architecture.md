# Model Architecture

## Overview

This project uses a deep 1D Convolutional Neural Network (CNN) designed for sequential signal analysis.

The model performs two tasks simultaneously:

1. **Anomaly Detection**
2. **Disease Classification**

It is implemented using TensorFlow and optimized for time-series biomedical signals.

---

# Input Shape

```python
(1000, 6)
```

- **1000** → sequence length / timesteps
- **6** → features/channels per timestep

Example:
- ECG leads
- sensor channels
- physiological measurements

---

# High-Level Architecture

```text
Input (1000,6)
        │
        ▼
Conv1D(64, kernel=7)
        │
BatchNorm
        │
MaxPool
        │
        ▼
Conv1D(128, kernel=5)
        │
BatchNorm
        │
MaxPool
        │
        ▼
Conv1D(256, kernel=3)
        │
BatchNorm
        │
MaxPool
        │
        ▼
SeparableConv1D(256, kernel=3)
        │
BatchNorm
        │
GlobalAveragePooling1D
        │
Dense(128)
        │
Dropout(0.4)
        │
 ┌───────────────┴───────────────┐
 │                               │
 ▼                               ▼
Anomaly Head              Disease Head
Dense(1,sigmoid)          Dense(4,softmax)
```

---

# Detailed Layer Breakdown

## 1. Input Layer

```python
inputs = tf.keras.Input(shape=(1000, 6))
```

Accepts multivariate sequential data.

---

## 2. First Convolution Block

```python
Conv1D(64, 7, padding='same', activation='relu')
```

Purpose:
- capture low-level temporal features
- detect broad signal patterns

Kernel Size:
- 7

Filters:
- 64

Followed by:

```python
BatchNormalization()
MaxPooling1D(2)
```

### Batch Normalization

Helps:
- stabilize training
- improve convergence
- reduce internal covariate shift

### MaxPooling

Reduces:
- temporal dimensionality
- computational cost

---

## 3. Second Convolution Block

```python
Conv1D(128, 5, padding='same', activation='relu')
```

Purpose:
- learn more complex temporal dependencies
- increase feature abstraction

Filters increased:
- 64 → 128

Followed by:
- BatchNormalization
- MaxPooling1D

---

## 4. Third Convolution Block

```python
Conv1D(256, 3, padding='same', activation='relu')
```

Purpose:
- capture highly abstract signal representations
- identify disease-specific patterns

Filters:
- 256

Smaller kernel:
- captures finer temporal details

Followed by:
- BatchNormalization
- MaxPooling1D

---

## 5. Depthwise Separable Convolution

```python
SeparableConv1D(256, 3, padding='same', activation='relu')
```

Purpose:
- reduce parameter count
- improve computational efficiency
- retain strong feature extraction

Followed by:
- BatchNormalization

---

## 6. Global Average Pooling

```python
GlobalAveragePooling1D()
```

Purpose:
- reduce feature maps into compact representations
- avoid excessive fully-connected parameters
- improve generalization

---

## 7. Dense Representation Layer

```python
Dense(128, activation='relu')
```

Learns high-level latent representations from extracted temporal features.

---

## 8. Dropout Regularization

```python
Dropout(0.4)
```

Purpose:
- reduce overfitting
- improve model robustness

Dropout Rate:
- 40%

---

# Multi-Task Output Heads

The architecture branches into two prediction heads.

---

## A. Anomaly Detection Head

```python
Dense(1, activation='sigmoid', name='anomaly')
```

Task:
- binary classification

Output:
- probability of anomaly

Activation:
- sigmoid

Loss:
```python
binary_crossentropy
```

---

## B. Disease Classification Head

```python
Dense(4, activation='softmax', name='disease')
```

Task:
- multi-class disease classification

Classes:
- 4 disease categories

Activation:
- softmax

Loss:
```python
categorical_crossentropy
```

---

# Multi-Task Learning

The model jointly learns:

1. anomaly detection
2. disease classification

---

# Optimizer

```python
Adam(
    learning_rate=1e-4,
    clipnorm=1.0
)
```

## Learning Rate

```python
1e-4
```

Chosen for:
- stable convergence
- smoother optimization

---

## Gradient Clipping

```python
clipnorm=1.0
```

Purpose:
- prevent exploding gradients
- stabilize deep sequential learning

---

# Loss Functions

```python
loss={
    'anomaly': 'binary_crossentropy',
    'disease': 'categorical_crossentropy'
}
```

---

# Metrics

```python
metrics={
    'anomaly': 'accuracy',
    'disease': 'categorical_accuracy'
}
```

Tracks:
- anomaly classification accuracy
- disease classification accuracy


# Why This Architecture Works Well

## 1D CNN Advantages

1D CNNs are effective for:
- ECG analysis
- sensor signals
- temporal biomedical patterns
- time-series anomaly detection

---

## Separable Convolution Advantages

Reduces:
- parameter count
- communication cost in federated learning

Important because federated learning repeatedly transmits model parameters between clients and server.

---

# Federated Learning Compatibility

This architecture is especially suitable for Federated Learning because:
- compact parameter size
- efficient local training
- reduced communication overhead
- lower client memory usage

---