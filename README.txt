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