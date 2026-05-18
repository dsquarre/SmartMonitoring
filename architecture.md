# Model Architecture

This model is a lightweight 1D CNN built using TensorFlow for sequential biomedical data of shape `(1000,6)`.

The network uses multiple `Conv1D` layers with increasing filters (64 → 128 → 256) to learn patterns from the signal data. Each convolution block is followed by Batch Normalization and MaxPooling for stable and efficient training.

A `SeparableConv1D` layer is used to reduce parameter count and make the model more efficient for federated learning environments.

After feature extraction, `GlobalAveragePooling1D` converts feature maps into compact representations, followed by a dense layer with dropout regularization.

The model has two output heads:
- `anomaly` → binary classification using sigmoid activation
- `disease` → 4 class classification using softmax activation

The model is trained using the Adam optimizer with gradient clipping (`clipnorm=1.0`) for stable training.

Loss functions:
- Binary Crossentropy for anomaly detection
- Categorical Crossentropy for disease classification

This architecture is lightweight, efficient and suitable for federated learning on distributed clients.