import torchaudio
import torch
import matplotlib.pyplot as plt
from pytube import YouTube
from pydub import AudioSegment
import os
import ffmpeg
import sys
import yt_dlp
import subprocess

def youtube_to_wav(video_url, output_path="data/dataset/temp_data/temp.wav"):
    try:
        # Download the audio using yt-dlp (output as .m4a or .webm)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'downloaded_audio.%(ext)s',  # Temporary file name
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from: {video_url}")
            ydl.download([video_url])

        # Load the downloaded audio with pydub (change extension based on what yt-dlp downloaded)
        audio = AudioSegment.from_file('downloaded_audio.webm')  # Adjust extension if needed

        # Export as .wav
        audio.export(output_path, format="wav")

        print(f"Audio successfully saved as {output_path}")

        # Cleanup downloaded audio
        os.remove(downloaded_audio.webm)

    except Exception as e:
        print(f"An error occurred: {e}")


youtube_to_wav("https://www.youtube.com/watch?v=NGPUnEewMxM")

# # Load audio
# audio_path = ''
# waveform, sample_rate = torchaudio.load(audio_path)

# # Create the MelSpectrogram transform
# mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
#     sample_rate=sample_rate,
#     n_mels=64,
#     f_min=125,
#     f_max=7500,
#     n_fft=1024,
#     hop_length=256
# )

# # Apply the transform
# mel_spectrogram = mel_spectrogram_transform(waveform)

# # Convert to decibels (log scale)
# log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

# # Plot the Mel spectrogram
# plt.figure(figsize=(10, 6))
# plt.imshow(log_mel_spectrogram[0].numpy(), aspect='auto', origin='lower', cmap='inferno')
# plt.title('Mel Spectrogram')
# plt.ylabel('Mel bins')
# plt.xlabel('Time')
# plt.colorbar(format="%+2.0f dB")
# plt.show()


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