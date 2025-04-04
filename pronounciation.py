from transformers import pipeline
import torchaudio
import torchaudio.transforms as T
import numpy as np

pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

audio_file = r"D:\GitHub\Presentation-Skills\audio\WhatsApp Audio 2024-10-21 at 8.55.53 PM.wav"


waveform, sample_rate = torchaudio.load(audio_file, format="wav")


if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

waveform = (waveform - waveform.mean()) / waveform.std()

waveform = waveform.squeeze(0).numpy()

result = pipe(waveform)

score_mapping = {
    "advanced": 100,
    "proficient": 75,
    "intermediate": 50,
    "beginer": 25  
}

final_score = sum(score_mapping[item["label"]] * item["score"] for item in result)

print(f"Pronunciation Score: {final_score:.2f} / 100")

