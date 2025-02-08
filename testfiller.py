import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Load the fine-tuned model and processor
num_labels = 6
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model.load_state_dict(torch.load("fine_tuned_wav2vec2.pth", map_location=torch.device("cpu")))
model.eval()

# Load and resample the audio file
audio_file = "audio/WhatsApp Audio 2024-10-21 at 8.55.53 PM.wav"
waveform, sample_rate = torchaudio.load(audio_file)
waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# Define chunking parameters
segment_duration = 2  # 2 seconds per chunk
sample_rate = 16000  # Wav2Vec2 expects 16kHz audio
samples_per_segment = segment_duration * sample_rate

total_duration = waveform.shape[1] / sample_rate  # Total duration in seconds
total_segments = total_duration // segment_duration  # Number of 2-sec chunks

# Process audio in 2-second chunks
filler_counts = {"Uh": 0, "Um": 0}
label_map = {0: "Uh", 1: "Words", 2: "Laughter", 3: "Um", 4: "Music", 5: "Breath"}

for start in range(0, waveform.shape[1], samples_per_segment):
    end = start + samples_per_segment
    segment = waveform[:, start:end]

    # Skip segments shorter than 2 seconds (last segment)
    if segment.shape[1] < samples_per_segment:
        break

    # Convert segment to model input
    inputs = processor(segment.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    # Get model predictions
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    # Count filler words
    for pred in predictions:
        label = label_map[pred.item()]
        if label in filler_counts:
            filler_counts[label] += 1

# Compute filler rate
total_fillers = filler_counts["Uh"] + filler_counts["Um"]
filler_rate = total_fillers / total_segments if total_segments > 0 else 0

# Convert to a 100-point score
score = max(100 - (filler_rate * 100), 0)

# Generate feedback
if score > 90:
    feedback = "Excellent! You used very few filler words. Keep it up! 🎯"
elif score > 75:
    feedback = "Good job! You have some fillers, but it's not distracting. Try to be more mindful. 😊"
elif score > 50:
    feedback = "You're using filler words often. Consider pausing instead of saying 'Uh' or 'Um'. 🧐"
else:
    feedback = "High use of filler words detected! Try slowing down and structuring your thoughts before speaking. 🚀"

# Print results
print(f"\n🎤 Speech Fluency Score: {score:.2f}/100")
print(f"📝 Feedback: {feedback}")
print(f"📊 Filler Word Breakdown: {filler_counts}")
