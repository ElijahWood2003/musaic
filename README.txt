---Written by Elijah Wood in 2025---
Co-Author: Ivan Fuentes

Musical Key-Signature Auto-Detector ML Model

Goal: Create a ML Model capable of continuous audio input key-signature detection.
Current model is being saved in git LFS.
Open in a Docker devcontainer for dependencies.


    Creating the Dataset
To create a large dataset with minimal manual labor, I have created a script which takes in a list of text in the format:

[YouTube-music URL],[key-signature] \ENTER
[YouTube-music URL],[key-signature] \ENTER
...
Through a series of subscripts, the script outputs this simple input into clean data.

Data Script Output (data_process.py, data_script.py):
1. Downloads and converts the Youtube video into a .wav audio file
2. Stores the .wav audio file in a local temp directory
3. Converts the .wav file into a spectrogram
4. Stores the spectrogram in a data directory
5. Adds the local path of the spectrogram to a .csv file along with the key-signature
6. Cleans up all temporary audio/files created


    Steps to Build the Model (audio_ml.py)
1. Preprocess the Data:
    Load the CSV file containing spectrogram paths and key signatures.
    Convert key-signatures into numerical labels

2. Load and Preprocess Spectrograms:
    Load spectrograms from their file paths.
    Split spectrograms into 'chunks' of smaller size to maximize training data
    Normalize pixel values (e.g., scale to [0, 1] or standardize).

3. Build the CNN Model:
    Use TensorFlow/Keras to create a CNN architecture.
    Includes convolutional layers, pooling layers, and fully connected layers.
    Adds dropout for regularization.

4. Compile the Model:
    Choose an optimizer -> using adam.
    Use a loss function suitable for classification -> using sparse_categorical_crossentropy since labels are ints.
    Add metrics -> currently just accuracy.

5. Train the Model:
    Use the training set to train the model.
    Validate the model on the validation set.

6. Evaluate the Model:
    Test the model on the test set.
    Analyze performance using the predefined metrics (accuracy).

7. Save and Deploy the Model:
    Save the trained model for future use.
    Saved in /models as a .keras file which is tracked by git-lfs


References:
https://www.tensorflow.org/api_docs/python/tf/keras/layers
https://www.tensorflow.org/api_docs/python/tf/keras/Model
