## CNN model to train off dataset
import tensorflow as tf
from keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

# relevant directory paths
music_data_dir = "data/dataset/music-data.csv"          # path to music-data.csv
models_dir = "models/"                                  # path to save model

# Defining functions for splitting the spectrograms into small chunks for training
# spectrogram = path to spectrogram png
# chunk_size = duration of each chunk in seconds; default is 3
# sr = sample rate of the original audio; default is 48000
# hop_length = hop length used to create the spectrogram; default is 512
def split_spectrogram(spectrogram, chunk_size=3, sr=48000, hop_length=512):
    # Calculate the number of time frames per chunk
    frames_per_chunk = int((chunk_size * sr) / hop_length)
    num_chunks = spectrogram.shape[1] // frames_per_chunk

    # Split the spectrogram into chunks
    chunks = [spectrogram[:, i * frames_per_chunk:(i + 1) * frames_per_chunk] for i in range(num_chunks)]
    return chunks


# loads spectrograms and splits them into smaller chunks; returns a list of these normalized chunks
def load_and_split_spectrogram(spectrogram_path, chunk_size=3, sr=48000, hop_length=512, img_size=(128, 128)):
    # load spectrogram image as a grayscale image (cv2.imread is used read .png and do this)
    spectrogram = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
    
    # normalize the spectrogram from [0, 255] -> [0, 1]
    spectrogram = spectrogram / 255.0

    # split the spectrogram into chunks
    chunks = split_spectrogram(spectrogram, chunk_size, sr, hop_length, img_size[0])

    # resize each chunk to the desired size
    resized_chunks = [cv2.resize(chunk, img_size) for chunk in chunks]

    # return chunks
    return resized_chunks


    # Load CSV file and encode key signatures
music_data = pd.read_csv(music_data_dir)
spectrogram_paths = music_data['spg_path'].values
key_signatures = music_data['ksig'].values

# Encode key signatures into numerical labels
label_encoder = LabelEncoder()
key_labels = label_encoder.fit_transform(key_signatures)

    # Process all spectrograms into chunks
X = []  # initial X npy array representing each chunk image
y = []  # initial Y npy array representing key signatures for each chunk
for spectrogram_path, key_label in zip(spectrogram_paths, key_labels):
    chunks = load_and_split_spectrogram(spectrogram_path, chunk_size=3)  # 3-second chunks
    X.extend(chunks)
    y.extend([key_label] * len(chunks)) # placing key signatures FOR EACH chunk

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Add a channel dimension for CNN input (additional input since we have grayscale [0, 1] input)
X = np.expand_dims(X, axis=-1)

    # Split dataset into train, validation, and test sets
# X_train = training chunks;          X_test = testing chunks
# y_train = training key signatures;  y_test = testing key signatures
# X_val = validation chunks;          y_val = validation key signatures
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Final splits:
#   Training set: 64% of original data
#   Validation set: 16% of original data
#   Test set: 20% of the original data


    # Build, compile and train
input_shape = X_train[0].shape              # input_shape = (128, 128, 1) for CNN model to know shape
num_classes = len(label_encoder.classes_)   # num_classes determines output estimate

# Creating the ML model
# Using a sequential model for a linear stack of layers (data flows from one stack to the next linearly)
# Each layer will use 'relu' which allows for non-linearity
model = models.Sequential([
    # First convolutional layer (128x128x1)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),  #  32 filters each with size 3x3
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimensions by half

    # Second convolutional layer (64x64x32)
    layers.Conv2D(64, (3, 3), activation='relu'),       # adds another layer with 64 filters; increasing filters allows model to learn more complex features
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimensions by another half

    # Third convolutional layer (32x32x64)
    layers.Conv2D(128, (3, 3), activation='relu'),      # adds another layer with 128 filters
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimension by another half; preparing data for fully connected layers

    # Flatten and fully connected layers
    layers.Flatten(),                                   # flattens the dimensions to prepare for feeding into dense layers
    layers.Dense(128, activation='relu'),               # adds a fully connected layer with 128 neurons; takes data from convolutional layers to maker predictions
    layers.Dropout(0.5),                                # randomly sets 50% of the input units to 0 at each update of training, helping with overfitting
    layers.Dense(num_classes, activation='softmax')     # a final dense layer based on the number of key signatures; each neuron is mapped to a unique key signature
])

model.compile(optimizer='adam',                             # adam = converges faster than traditional optimizers like Stochastic Gradient Descent
              loss='sparse_categorical_crossentropy',       # loss = 'cost' we want to minimize
              metrics=['accuracy'])                         # accuracy = trains based on percentage of predictions matching true labels

history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val),
                    batch_size=32)


# 5. Evaluating and saving model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save(f'{models_dir}key_signature_model.h5')