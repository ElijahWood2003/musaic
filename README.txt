---Written by Elijah Wood in 2025---
Musical Key-Signature Auto-Detector ML Model


    Creating the Dataset
To create a large dataset with minimal manual labor, I have created a script which takes in a list of text in the format:

[YouTube-music URL], [key-signature] \ENTER
[YouTube-music URL], [key-signature] \ENTER
...
Through a series of subscripts, the script outputs this simple input into clean data.

Data Script Output (data_process, data_script):
1. Downloads and converts the Youtube video into a .wav audio file
2. Stores the .wav audio file in a local temp directory
3. Converts the .wav file into a spectrogram
4. Stores the spectrogram in a data directory
5. Adds the local path of the spectrogram to a .csv file along with the key-signature
6. Cleans up all temporary audio/files created
7. audio_ml: CNN model to train off of our dataset


    Steps to Build the Model
1. Preprocess the Data:
    Load the CSV file containing spectrogram paths and key signatures.
    Split the dataset into training, validation, and test sets.
    Convert key-signatures into numerical labels

2. Load and Preprocess Spectrograms:
    Load spectrograms from their file paths.
    Split spectrograms into 'chunks' of smaller size to maximize training data
    Normalize pixel values (e.g., scale to [0, 1] or standardize).

3. Build the CNN Model:
    Use TensorFlow/Keras to create a CNN architecture.
    Include convolutional layers, pooling layers, and fully connected layers.
    Add dropout or batch normalization for regularization.

4. Compile the Model:
    Choose an optimizer (e.g., Adam).
    Use a loss function suitable for classification (e.g., sparse_categorical_crossentropy if labels are integers).
    Add metrics like accuracy.

5. Train the Model:
    Use the training set to train the model.
    Validate the model on the validation set.
    Monitor for overfitting and adjust hyperparameters if necessary.

6. Evaluate the Model:
    Test the model on the test set.
    Analyze performance using metrics like accuracy, confusion matrix, etc.

7. Save and Deploy the Model:
    Save the trained model for future use.
    Optionally, deploy the model for inference.