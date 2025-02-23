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

# relevant values to be messed with in training
CHUNK_SIZE = 3      # chunk size (in seconds)
BATCH_SIZE = 30     # batch size -> # of samples tested before updating gradient (smaller the number the more processing power but potentially better results)
NUM_EPOCHS = 10      # number of times we iterate through the entire set of test samples
IMG_SCALE = 1       # multiplied by img_size of chunks -> 0.5 halves the chunk image size; when IMG_SCALE = 1 the chunk maintains all data


# Defining functions for splitting the spectrograms into small chunks for training
# spectrogram = path to spectrogram png
# chunk_size = duration of each chunk in seconds; default is 3
# sr = sample rate of the original audio; default is 48000; all sr's should be equal
# hop_length = hop length used to create the spectrogram; default is 512
def split_spectrogram(spectrogram, chunk_size=3, sr=48000, hop_length=512):
    # Calculate the number of time frames per chunk
    frames_per_chunk = int((chunk_size * sr) / hop_length)
    num_chunks = spectrogram.shape[1] // frames_per_chunk

    # Split the spectrogram into chunks
    chunks = [spectrogram[:, i * frames_per_chunk : (i + 1) * frames_per_chunk] for i in range(num_chunks)]
    return chunks, frames_per_chunk


# loads spectrograms and splits them into smaller chunks; returns a list of these normalized chunks
def load_and_split_spectrogram(spectrogram_path, height, chunk_size=3, sr=48000, hop_length=512):
    # load spectrogram image as a grayscale image
    spectrogram = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
    
    # normalize the spectrogram from [0, 255] -> [0, 1]
    spectrogram = spectrogram / 255.0

    # split the spectrogram into chunks
    chunks, frames_per_chunk = split_spectrogram(spectrogram, chunk_size, sr, hop_length)

    # setting image size; when IMG_SCALE = 1 the chunk maintains all data
    img_size = (int(frames_per_chunk * IMG_SCALE), int(height * IMG_SCALE))

    # resizes based on img_size (IMG_SCALE)
    resized_chunks = [cv2.resize(chunk, img_size) for chunk in chunks]

    # return chunks
    return chunks, frames_per_chunk


    # Load CSV file data and encode key signatures
music_data = pd.read_csv(music_data_dir)
spectrogram_paths = music_data['spg_path'].values
key_signatures = music_data['ksig'].values
sample_rates = music_data['sample_rate'].values
heights = music_data['height'].values

# Encode key signatures into numerical labels
label_encoder = LabelEncoder()
key_labels = label_encoder.fit_transform(key_signatures)
frames_per_chunk = 0

    # Process all spectrograms into chunks
X = []  # initial X npy array representing each chunk image
y = []  # initial Y npy array representing key signatures for each chunk
for spectrogram_path, key_label, sample_rate, height in zip(spectrogram_paths, key_labels, sample_rates, heights):
    chunks, frames_per_chunk = load_and_split_spectrogram(spectrogram_path, height, chunk_size=CHUNK_SIZE, sr=sample_rate)

    # adding x and y data based on chunks
    X.extend(chunks)
    y.extend([key_label] * len(chunks)) # placing key signatures FOR EACH chunk

# Process message
print("Processed spectrograms into chunks.\n")

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
# initial size of image -> frames_per_chunk ~= 281; height = 1200
model = models.Sequential([
    # First convolutional layer; initial dim: (281x1200x1)
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),  #  16 filters each with size 281x1200
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimensions by half

    # Second convolutional layer; initial dim: (140x600x16)
    layers.Conv2D(32, (3, 3), activation='relu'),       # adds another layer with 32 filters; increasing filters allows model to learn more complex features
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimensions by another half

    # Third convolutional layer; initial dim: (70x300x32)
    layers.Conv2D(64, (3, 3), activation='relu'),       # adds another layer with 64 filters
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimension by another half

    # Fourth convolutional layer; initial dim: (35x150x64)
    layers.Conv2D(64, (3, 3), activation='relu'),       # adds another layer with 128 filters
    layers.MaxPooling2D((2, 2)),                        # reduces the spatial dimension by another half; preparing data for fully connected layers

    # Flatten and fully connected layers
    layers.Flatten(),                                   # flattens the dimensions to prepare for feeding into dense layers
    layers.Dense(128, activation='relu'),               # adds a fully connected layer with 128 neurons; takes data from convolutional layers to make predictions
    layers.Dropout(0.5),                                # randomly sets 50% of the input units to 0 at each update of training, helping with overfitting
    layers.Dense(num_classes, activation='softmax')     # a final dense layer based on the number of key signatures; each neuron is mapped to a unique key signature
])

# compiles and configures the model for training
model.compile(optimizer='adam',                             # adam = converges faster than traditional optimizers like Stochastic Gradient Descent
              loss='sparse_categorical_crossentropy',       # loss = 'cost' we want to minimize
              metrics=['accuracy'])                         # accuracy = trains based on percentage of predictions matching true labels

# trains the model for a fixed number of dataset iterations / epochs
history = model.fit(X_train, y_train, epochs=NUM_EPOCHS,    # X_train / y_train: input -> correct output of training data; epochs = # of times we iterate over entire dataset
                    validation_data=(X_val, y_val),         # validation data: the x and y val data to evaluate loss after each epoch
                    batch_size=BATCH_SIZE)                  # batch size: number of samples tested before a gradient update


    # Evaluating and saving model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the model
model.save(f'{models_dir}key_signature_model_v0.keras')