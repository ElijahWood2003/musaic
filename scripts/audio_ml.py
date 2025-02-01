## CNN model to train off dataset
import tensorflow as tf
from keras import layers, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

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