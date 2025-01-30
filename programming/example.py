# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:01:06 2025

@author: aron-
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_data():
    np.random.seed(42)
    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 3 * x + np.random.normal(0, 0.1, x.shape)
    return x, y

# Prepare the dataset
x, y = generate_data()
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

# Build the model
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(1,), name='Hidden_Layer_1'),
    layers.Dense(8, activation='relu', name='Hidden_Layer_2'),
    layers.Dense(8, activation='relu', name='Hidden_Layer_3'),
    layers.Dense(1, name='Output_Layer')
])

# Visualize the model structure
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(dataset, epochs=50, verbose=1)

# Evaluate the model
test_x = np.array([[-0.5], [0.0], [0.5]])
predictions = model.predict(test_x)

# Visualization of Model Metrics
plt.figure(figsize=(12, 5))

# Plot the Loss over epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# Plot the Mean Absolute Error over epochs
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Mean Absolute Error (MAE)', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training MAE Over Epochs')
plt.legend()

plt.show()

# Visualizing the Predictions vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Model Prediction')
plt.scatter(test_x, predictions, color='green', label='Test Predictions', zorder=5)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Model Predictions vs Actual Data')
plt.legend()
plt.show()

print("\nPredictions for test inputs:")
for i, pred in enumerate(predictions):
    print(f"Input: {test_x[i][0]:.2f}, Prediction: {pred[0]:.2f}")
