import torchaudio
import torch
import matplotlib.pyplot as plt
from pytube import YouTube
from pydub import AudioSegment
import os
import sys
import yt_dlp
import subprocess
import pandas as pd
import cv2
from PIL import Image
# import ffmpeg -> if needed i can use ffmpeg-python in requirements.txt

# all necessary file directories
unprocessed_data_dir = "data/unprocessed-data.csv"      # path to unprocessed-data.csv
music_data_dir = "data/dataset/music-data.csv"          # path to music-data.csv
temp_wav_dir = "data/temp_data/temp.wav"                # path for temp wav file
spectrogram_dir = "data/dataset/spectrograms"           # path for spectrograms


# combines all scripts below to take unprocessed-data.csv -> music-data.csv with paths to spectrograms and ksig
# empties unprocessed-data.csv; skips and marks any URLs that fail
def process_data():
    # get data frames
    ud_df = pd.read_csv(unprocessed_data_dir)
    md_df = pd.read_csv(music_data_dir)

    processed_rows = []    # indexes of processed rows in the csv file to drop later
    index = 0              # tracking indices of unprocessed-data to know which we processed
    abs_index = len(md_df)     # tracking absolute index of music-data for naming spectrograms
    
    # iterating through rows with itertuples
    for row in ud_df.itertuples():
        # yt / ksig values from the row of df
        yt_url = row.URL
        ksig = row.ksig
        video_title = ""

        # generating .wav file from yt link
        video_title = youtube_to_wav(yt_url)
        if(video_title == None):
            print(f"Failed to generate spectrogram for {yt_url}\n")
            index += 1
            continue
        
        # generating spectrogram from .wav file and holding onto the path
        spg_path, sample_rate, width, height = wav_to_spectrogram(abs_index)
        
        # check spg_path exists
        if(spg_path == None or os.path.exists(spg_path) == False):
            print(f"Failed to generate spectrogram for {yt_url}\n")
            index += 1
            continue

        # place information into music-data.csv if successful
        md_df.loc[len(md_df)] = [f'{spg_path}',f'{ksig}', f'{sample_rate}', f'{width}', f'{height}', f'{yt_url}', f'{video_title}']
        
        processed_rows.append(index)
        index += 1
        abs_index += 1

        print("")
        
    # dropping rows we have processed
    ud_df = ud_df.drop(index=processed_rows)
    ud_df = ud_df.reset_index(drop=True)
    
    # if any data remains unprocessed this means some error occured while processing it
    # ask the user if they would like to delete or keep the unprocessed data
    unprocessed = len(ud_df)
    if(unprocessed > 0):
        inp = input(f"There is {unprocessed} unprocessed data. Would you like to delete? (y/n) ")
        if(inp == "Y" or inp == "y"):
            ud_df = ud_df.iloc[0:0]
    
    # saving our data frames to csv files
    ud_df.to_csv(unprocessed_data_dir, index=False, header=True)
    md_df.to_csv(music_data_dir, index=False, header=True)

# takes a youtube url and creates a .wav audio file in a temp directory
# returns title of video when successful; None otherwise
def youtube_to_wav(video_url, output_path=temp_wav_dir) -> str | None:
    try:
        video_title = ""

        # Download the audio using yt-dlp (output as .m4a or .webm)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',  # Temporary file name
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {video_url}")
            video_title = ydl.extract_info(video_url, download=False).get('title', None)
            ydl.download([video_url])

        # Check to make sure temp audio is .webm (could be m4a file)
        if(os.path.exists('downloaded_audio.m4a')):
            os.remove("downloaded_audio.m4a")
            return None

        # Load the downloaded audio with pydub (change extension based on what yt-dlp downloaded)
        audio = AudioSegment.from_file('downloaded_audio.webm')  # Adjust extension if needed

        # Export as .wav
        audio.export(output_path, format="wav")

        print(f"Audio successfully saved as {output_path}")

        # Cleanup downloaded audio
        os.remove("downloaded_audio.webm")
        return video_title

    # any error we encounter along the way just return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Load temp audio -> saves it to spectrograms data and returns path directory to spectrogram
# file_name = path to .wav file
# index = index value to name spg in directory
def wav_to_spectrogram(index):
    # audio path for temp wav file
    audio_path = temp_wav_dir

    # checking the path exists; fails if it doesn't
    if(os.path.exists(audio_path) == False):
        return None

    # loading waveform and sample rate from audio
    waveform, sample_rate = torchaudio.load(audio_path, format="wav")

    # Create the MelSpectrogram transform
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        
            # number of mel bands
        n_mels=64,
        
            # defines the frequency range of interest
        f_min=125,
        f_max=7500,
        
            # n_fft = number of samples per segment
            # a larger value gives better frequency resolution but worse time resolution
        n_fft=1024,      
        
            # defines the number of samples between successive windows
            # reducing it increases resolution but also increases computation cost
            # generally hop_length = 1/4 * n_fft
        hop_length=256
    )

    # Apply the transform
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Convert to decibels (log scale)
    log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    spectrogram = log_mel_spectrogram[0] # (num of mel bins, num of channels)

    # Creating name for new spectrogram
    df = pd.read_csv(music_data_dir)
    
    # Define the output path
    output_dir = spectrogram_dir
    output_name = f"sp_{index}.png"
    os.makedirs(output_dir, exist_ok=True)
    output_str = f"{output_dir}/{output_name}"      # need this string to return
    output_path = os.path.join(output_dir, output_name)

    # Save the spectrogram as an image
    plt.figure(figsize=(10, 4))  # Adjust the size for consistent image dimensions
    plt.imshow(spectrogram.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
    plt.axis("off")  # Remove axes for a clean image
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()  # Close the figure to free memory

    # getting height / width in pixels
    image = Image.open(output_path)
    width, height = image.size
    image.close()

    print(f"Spectrogram saved as an image at {output_path}")

    # Cleanup temp_data audio
    os.remove(audio_path)
    
    # return output path to store in csv
    return output_str, sample_rate, width, height


# reprocesses data (FROM music-data.csv)
# TAKES ALL URLS/KEY-SIGS FROM MUSIC-DATA, DELETES SPECTROGRAMS AND REPROCESSES (EXPENSIVE)
def reprocess_data():
    # double checking user understands and wants to proceed
    inp = input(f"ARE YOU SURE YOU WOULD LIKE TO REPROCESS ALL DATA? THIS WILL DELETE ALL SPECTROGRAMS IN DATA AND REPROCESS THEM (y/n) ")
    if(inp == "Y" or inp == "y"):
        # df we want access to
        md_df = pd.read_csv(music_data_dir)
        ud_df = pd.read_csv(unprocessed_data_dir)

        # getting values from music data
        spectrogram_paths = md_df['spg_path'].values
        key_signatures = md_df['ksig'].values
        urls = md_df['URL'].values

        # iterating through lists; 
        # placing url / ksig in unprocessed data AND removing spectrogram AND dropping row of music data
        for spectrogram_path, key_signature, url, i in zip(spectrogram_paths, key_signatures, urls, range(len(md_df))):
            ud_df.loc[len(ud_df)] = [f'{url}', f'{key_signature}']
            os.remove(spectrogram_path)
            md_df = md_df.drop(index=i)
        
        # placing new df's into respective .csv files
        ud_df.to_csv(unprocessed_data_dir, index=False, header=True)
        md_df.to_csv(music_data_dir, index=False, header=True)

        # running process_data
        process_data()

    else:
        print("Reprocess data canceled.\n")


# parse dataset to output information about it
def print_data_info():
    df = pd.read_csv(music_data_dir)

    # a dict representing the number of each key sig in the dataset
    ksig_dict = {
        "cmajor": 0,
        "c#major": 0,
        "dmajor": 0,
        "d#major": 0,
        "emajor": 0,
        "fmajor": 0,
        "f#major": 0,
        "gmajor": 0,
        "g#major": 0,
        "amajor": 0,
        "a#major": 0,
        "bmajor": 0,
        "cminor": 0,
        "c#minor": 0,
        "dminor": 0,
        "d#minor": 0,
        "eminor": 0,
        "fminor": 0,
        "f#minor": 0,
        "gminor": 0,
        "g#minor": 0,
        "aminor": 0,
        "a#minor": 0,
        "bminor": 0
    }

    ksig = df['ksig'].values

    for key in ksig:
        ksig_dict[key] += 1

    # print out information in dict
    print(ksig_dict)

        ## ESTIMATING KEY SIGNATURE WITH ESSENTIA (~80% ACCURATE)
# import librosa
# from essentia.standard import KeyExtractor

# # Load audio with librosa
# audio_path = "data/genres_original/blues/blues.00000.wav"
# y, sr = librosa.load(audio_path, sr=None)

# # Extract the key using Essentia
# key_extractor = KeyExtractor()
# key, scale, strength = key_extractor(y)

# print(f"Estimated Key: {key}")
# print(f"Scale: {scale}")
# print(f"Strength: {strength}")
