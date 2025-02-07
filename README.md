# Handwritten Character Recognition(part1)

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow/Keras for recognizing handwritten characters using the EMNIST ByClass dataset.

## Dataset

The EMNIST ByClass dataset is a set of handwritten character images, including both uppercase and lowercase letters, as well as digits. The dataset is publicly available and can be downloaded from the following link:

[Download EMNIST ByClass Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Project Structure

- `Task1_part1.py`: py file containing the complete code for data preprocessing, model training, evaluation, and visualization.
- `Task1_part2.py`: py file containing the complete code for data preprocessing, model training, evaluation, and visualization.
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
for me it I uploaded to my google drive 

/content/drive/MyDrive/Colab_Notebooks/Datasets/EMNIST/

## Code Overview

## Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

## Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import string

## Load Dataset
mat_file_path = '/content/drive/MyDrive/Colab_Notebooks/Datasets/EMNIST/emnist-byclass.mat'
emnist_data = sio.loadmat(mat_file_path)

## Data Preprocessing
### Extract training images and labels
X_train = emnist_data['dataset'][0][0][0][0][0][0]
y_train = emnist_data['dataset'][0][0][0][0][0][1]

### Extract testing images and labels
X_test = emnist_data['dataset'][0][0][1][0][0][0]
y_test = emnist_data['dataset'][0][0][1][0][0][1]

### Reshape and normalize images
X_train = X_train.reshape(-1, 28, 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28).astype('float32') / 255.0

### Transpose and flip images for correct orientation
X_train = np.flip(np.transpose(X_train, (0, 2, 1)), axis=2).reshape(-1, 28, 28, 1)
X_test = np.flip(np.transpose(X_test, (0, 2, 1)), axis=2).reshape(-1, 28, 28, 1)

### Flatten labels
y_train = y_train.flatten().astype(np.int64)
y_test = y_test.flatten().astype(np.int64)

### Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

## Model Definition
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

## Model Compilation and Training
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=15, batch_size=256)

## Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')

## Visualization
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

## Prediction
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

# Handwritten Words Recognition(part2)

This repository contains a TensorFlow/Keras implementation for recognizing handwritten text using the IAM Handwriting Database. The code includes data preprocessing, model training, evaluation, and visualization.

## Dataset

The dataset is downloaded and loaded from the code so no need to download any dataset beforehand

## Dependencies

To run this project, you need to have the following libraries installed:

- numpy
- matplotlib
- tensorflow

## Code Overview

## Dataset Acquisition
!wget -q https://github.com/sayakpaul/Handwriting-Recognizer-in-Keras/releases/download/v1.0.0/IAM_Words.zip
!unzip -qq IAM_Words.zip

### Create directories to store the data
!mkdir handwriting_data
!mkdir handwriting_data/samples
!tar -xf IAM_Words/words.tgz -C handwriting_data/samples
!mv IAM_Words/words.txt handwriting_data

### Display the first 20 lines of the words.txt file
!head -20 handwriting_data/words.txt

## Essential Imports
from tensorflow.keras.layers import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
### Set random seeds for reproducibility
np.random.seed(2023)
tf.random.set_seed(2023)

## Dataset Partitioning
dataset_root = "handwriting_data"
entries = []

### Read the words.txt file
raw_data = open(f"{dataset_root}/words.txt", "r").readlines()
for entry in raw_data:
    if entry[0] == "#":
        continue  # Skip comment lines
    if entry.split(" ")[1] != "err":  # Skip invalid entries
        entries.append(entry)

print(f"Total valid entries: {len(entries)}")

### Shuffle the entries
np.random.shuffle(entries)

### Create splits: 85% train, 7.5% validation, 7.5% test
train_split = int(0.85 * len(entries))
train_data = entries[:train_split]
remaining = entries[train_split:]

val_split = int(0.5 * len(remaining))
validation_data = remaining[:val_split]
test_data = remaining[val_split:]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(validation_data)}")
print(f"Testing samples: {len(test_data)}")

## Data Processing Pipeline
samples_dir = os.path.join(dataset_root, "samples")

### Function to load sample data
def load_sample_data(data_entries):
    image_paths = []
    processed_entries = []
    for entry in data_entries:
        components = entry.strip().split(" ")
        filename = components[0]

        # Extract directory structure components
        dir_part1 = filename.split("-")[0]
        dir_part2 = f"{dir_part1}-{filename.split('-')[1]}"
        full_path = os.path.join(
            samples_dir,
            dir_part1,
            dir_part2,
            f"{filename}.png"
        )

        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            image_paths.append(full_path)
            processed_entries.append(components[-1].strip())

    return image_paths, processed_entries

### Load the training, validation, and testing data
train_images, train_texts = load_sample_data(train_data)
val_images, val_texts = load_sample_data(validation_data)
test_images, test_texts = load_sample_data(test_data)

## Character Vocabulary Setup
unique_chars = set()
max_text_length = 0

### Analyze training texts for unique characters and maximum text length
for text in train_texts:
    unique_chars.update(list(text))
    max_text_length = max(max_text_length, len(text))

unique_chars = sorted(list(unique_chars))
vocab_size = len(unique_chars)

print(f"Longest text sequence: {max_text_length}")
print(f"Character vocabulary: {vocab_size}")

### Text cleaning for validation and test sets
def process_text_labels(labels):
    return [label.split(" ")[-1].strip() for label in labels]

### Clean validation and test texts
val_texts = process_text_labels(val_texts)
test_texts = process_text_labels(test_texts)

### Character Encoding Utilities
char_encoder = StringLookup(
    vocabulary=unique_chars,
    mask_token=None
)
char_decoder = StringLookup(
    vocabulary=char_encoder.get_vocabulary(),
    invert=True,
    mask_token=None
)

## Image Preprocessing
def preserve_aspect_resize(image, target_dims):
    target_w, target_h = target_dims
    image = tf.image.resize(image, (target_h, target_w), preserve_aspect_ratio=True)

### Calculate required padding
    pad_vert = target_h - tf.shape(image)[0]
    pad_horz = target_w - tf.shape(image)[1]

### Apply symmetric padding
    image = tf.pad(
        image,
        [
            [pad_vert // 2, pad_vert - pad_vert // 2],
            [pad_horz // 2, pad_horz - pad_horz // 2],
            [0, 0]
        ]
    )

    return image

## Data Pipeline Configuration
BATCH_SIZE = 72
IMG_WIDTH = 144
IMG_HEIGHT = 36
PAD_TOKEN = 0  # Padding token for text

### Function to prepare images
def prepare_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = preserve_aspect_resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return tf.cast(image, tf.float32) / 255.0

### Function to encode text labels
def encode_text(label):
    encoded = char_encoder(tf.strings.unicode_split(label, "UTF-8"))
    padding = max_text_length - tf.shape(encoded)[0]
    return tf.pad(encoded, [[0, padding]], constant_values=PAD_TOKEN)

### Function to create dataset
def create_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(
        lambda img, lbl: {
            "image": prepare_image(img),
            "label": encode_text(lbl)
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

### Generate datasets
training_ds = create_dataset(train_images, train_texts)
validation_ds = create_dataset(val_images, val_texts)
testing_ds = create_dataset(test_images, test_texts)

## Data Visualization
def display_samples(dataset, num_samples=8):
    data_batch = next(iter(dataset.take(1)))
    images = data_batch["image"]
    labels = data_batch["label"]

    plt.figure(figsize=(12, 9))
    for i in range(num_samples):
        ax = plt.subplot(3, 3, i + 1)

### Process image
        img = images[i].numpy()
        img = np.squeeze(img, axis=-1)
        img = (img * 255).astype(np.uint8)

### Decode label
        label = labels[i]
        label_chars = [c for c in label if c != PAD_TOKEN]
        decoded_text = tf.strings.reduce_join(
            char_decoder(label_chars)
        ).numpy().decode()

        ax.imshow(img, cmap="viridis")
        ax.set_title(decoded_text)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

### Display samples from the training dataset
display_samples(training_ds)
