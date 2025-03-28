from transformers import pipeline
import torchaudio
import torchaudio.transforms as T
import numpy as np

# Load the model
pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Path to your local audio file
audio_file = r"D:\GitHub\Presentation-Skills\audio\WhatsApp Audio 2024-10-21 at 8.55.53 PM.wav"


# Load the audio
waveform, sample_rate = torchaudio.load(audio_file, format="wav")


# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed
if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Convert tensor to numpy array
waveform = waveform.squeeze(0).numpy()

# Run prediction
result = pipe(waveform)

# Define score mapping
score_mapping = {
    "advanced": 100,
    "proficient": 75,
    "intermediate": 50,
    "beginer": 25  # Note: 'beginer' is misspelled in the model output, so keeping it as is
}

# Compute final score
final_score = sum(score_mapping[item["label"]] * item["score"] for item in result)

# Print result
print(f"Pronunciation Score: {final_score:.2f} / 100")

