import torchaudio
import torch
import matplotlib.pyplot as plt

# Load audio
audio_path = 'data/ColoredEyes.wav'
waveform, sample_rate = torchaudio.load(audio_path)

# Create the MelSpectrogram transform
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_mels=64,
    f_min=125,
    f_max=7500,
    n_fft=1024,
    hop_length=256
)

# Apply the transform
mel_spectrogram = mel_spectrogram_transform(waveform)

# Convert to decibels (log scale)
log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

# Plot the Mel spectrogram
# plt.figure(figsize=(10, 6))
# plt.imshow(log_mel_spectrogram[0].numpy(), aspect='auto', origin='lower', cmap='inferno')
# plt.title('Mel Spectrogram')
# plt.ylabel('Mel bins')
# plt.xlabel('Time')
# plt.colorbar(format="%+2.0f dB")
# plt.show()
