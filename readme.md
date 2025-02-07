# Handwritten Character Recognition part 1

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow/Keras for recognizing handwritten characters using the EMNIST ByClass dataset.

## Dataset

The EMNIST ByClass dataset is a set of handwritten character images, including both uppercase and lowercase letters, as well as digits. The dataset is publicly available and can be downloaded from the following link:

[Download EMNIST ByClass Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Project Structure

- `Task1.py`: py file containing the complete code for data preprocessing, model training, evaluation, and visualization.
- `README.md`: This README file.

## Dependencies

To run this project, you need to have the following libraries installed:

- numpy
- matplotlib
- scipy
- tensorflow
- google.colab (if running on Google Colab)

## Setup Instructions

Download the EMNIST ByClass dataset and place the emnist-byclass.mat file in the appropriate directory:


/content/drive/MyDrive/Colab_Notebooks/Datasets/EMNIST/
Open the Task1.ipynb notebook in your preferred environment (e.g., Google Colab).

Code Overview
Mount Google Drive
Mount Google Drive to access the dataset:

python
from google.colab import drive
drive.mount('/content/drive')
Import Libraries
Import the necessary libraries:

python
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import string
Load Dataset
Load the EMNIST ByClass dataset:

python
mat_file_path = '/content/drive/MyDrive/Colab_Notebooks/Datasets/EMNIST/emnist-byclass.mat'
emnist_data = sio.loadmat(mat_file_path)
Data Preprocessing
Preprocess the data:

python
# Extract training images and labels
X_train = emnist_data['dataset'][0][0][0][0][0][0]
y_train = emnist_data['dataset'][0][0][0][0][0][1]

# Extract testing images and labels
X_test = emnist_data['dataset'][0][0][1][0][0][0]
y_test = emnist_data['dataset'][0][0][1][0][0][1]

# Reshape and normalize images
X_train = X_train.reshape(-1, 28, 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28).astype('float32') / 255.0

# Transpose and flip images for correct orientation
X_train = np.flip(np.transpose(X_train, (0, 2, 1)), axis=2).reshape(-1, 28, 28, 1)
X_test = np.flip(np.transpose(X_test, (0, 2, 1)), axis=2).reshape(-1, 28, 28, 1)

# Flatten labels
y_train = y_train.flatten().astype(np.int64)
y_test = y_test.flatten().astype(np.int64)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
Model Definition
Define the CNN model:

python
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
Model Compilation and Training
Compile and train the model:

python
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=15, batch_size=256)
Model Evaluation
Evaluate the model on the test data:

python
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')
Visualization
Plot training and validation accuracy and loss:

python
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
Prediction
Function to predict and display the label for a given sample index:

python
def predict_and_display(sample_index):
    image = X_test[sample_index]
    true_label = y_test[sample_index]
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {emnist_labels[predicted_label]}, Actual: {emnist_labels[true_label]}')
    plt.axis('off')
    plt.show()
    print(f'Predicted Label: {emnist_labels[predicted_label]}, Actual Label: {emnist_labels[true_label]}')

sample_indices = [10, 200, 600, 1200, 3200]
for idx in sample_indices:
    predict_and_display(idx)
