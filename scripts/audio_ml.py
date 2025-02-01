## CNN model to train off dataset
import tensorflow as tf
from keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2


# Defining functions for splitting the spectrograms into small chunks for training
def split_spectrogram(spectrogram, chunk_size=3, sr=22050, hop_length=512):
    """
    Split a spectrogram into smaller chunks.
    - `chunk_size`: Duration of each chunk in seconds.
    - `sr`: Sample rate of the original audio.
    - `hop_length`: Hop length used to create the spectrogram.
    Returns a list of spectrogram chunks.
    """
    # Calculate the number of time frames per chunk
    frames_per_chunk = int((chunk_size * sr) / hop_length)
    num_chunks = spectrogram.shape[1] // frames_per_chunk

    # Split the spectrogram into chunks
    chunks = [spectrogram[:, i * frames_per_chunk:(i + 1) * frames_per_chunk] for i in range(num_chunks)]
    return chunks

def load_and_split_spectrogram(spectrogram_path, chunk_size=3, sr=22050, hop_length=512, img_size=(128, 128)):
    """
    Load a spectrogram and split it into smaller chunks.
    Returns a list of resized and normalized spectrogram chunks.
    """
    spectrogram = np.load(spectrogram_path)  # Assuming spectrograms are saved as .npy files
    chunks = split_spectrogram(spectrogram, chunk_size, sr, hop_length)
    resized_chunks = [cv2.resize(chunk, img_size) for chunk in chunks]  # Resize each chunk
    normalized_chunks = [(chunk - chunk.min()) / (chunk.max() - chunk.min()) for chunk in resized_chunks]  # Normalize
    return normalized_chunks


    # 3. Update the dataset
# Load CSV file
data = pd.read_csv('dataset.csv')  # Replace with your CSV file path
spectrogram_paths = data['spectrogram_path'].values
key_signatures = data['key_signature'].values

# Encode key signatures into numerical labels
label_encoder = LabelEncoder()
key_labels = label_encoder.fit_transform(key_signatures)

# Process all spectrograms into chunks
X = []
y = []
for spectrogram_path, key_label in zip(spectrogram_paths, key_labels):
    chunks = load_and_split_spectrogram(spectrogram_path, chunk_size=3)  # 3-second chunks
    X.extend(chunks)
    y.extend([key_label] * len(chunks))

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add a channel dimension for CNN input
X = np.expand_dims(X, axis=-1)

# Split dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # 4. Build, compile and train
input_shape = X_train[0].shape
num_classes = len(label_encoder.classes_)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val),
                    batch_size=32)


# 5. Evaluating and saving model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save('key_signature_model_spectrogram_chunks.h5')