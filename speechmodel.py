import speech_recognition as sr
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import difflib
import numpy as np
import spacy
import string
import aifc
import os
from moviepy import VideoFileClip
import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import pipeline
import torchaudio
import torchaudio.transforms as T


# Function to extract audio from video and save it as .wav
def extract_audio_from_video(video_file_path):
    # Get the base name of the video file (without extension)
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    
    # Generate the audio file path by changing the extension to .wav
    audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
    
    # Extract audio from the video
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)  # Save audio as .wav file
    
    return audio_file_path

# Specify your video file path
video_file_path = r"D:\GitHub\Presentation-Skills\Videos\TedTalk.mp4"

# Automatically generate the audio file path and extract the audio
audio_file_path = extract_audio_from_video(video_file_path)

# Print the generated audio file path
print(f"Audio extracted and saved at: {audio_file_path}")

# Initialize recognizer class (for recognizing speech)
recognizer = sr.Recognizer()

# Load the audio file using pydub to get duration
audio_segment = AudioSegment.from_file(audio_file_path)

# Get the duration of the audio file in seconds
audio_duration = len(audio_segment) / 1000  # pydub returns duration in milliseconds

# Load the audio file for the speech recognizer
audio_file = sr.AudioFile(audio_file_path)

# Process the audio file in chunks
with audio_file as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)  # Optional: helps with noisy audio

    # Define chunk size in seconds (e.g., 30 seconds per chunk)
    chunk_size = 30

    transcription = ""  # To store the full transcription
    current_position = 0

    # Loop through the audio in chunks
    while current_position < audio_duration:
        try:
            # Record the next chunk
            audio_chunk = recognizer.record(source, duration=chunk_size)

            # Transcribe the chunk
            chunk_text = recognizer.recognize_google(audio_chunk)
            transcription += chunk_text + " "  # Append chunk transcription
            print(f"Transcribed chunk {current_position}-{current_position+chunk_size}s: {chunk_text}")
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand the audio between {current_position}s and {current_position+chunk_size}s")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        # Update position to move to the next chunk
        current_position += chunk_size

# Load the pre-trained T5 model for grammar correction
model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

def correct_grammar(sentence, max_length=512):
    corrected_text = ""
    i = 0
    while i < len(sentence):
        # Get the next chunk of the sentence
        chunk = sentence[i:i + max_length]
        # Prepare input with prefix "grammar: " for T5 model
        input_text = "grammar: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

        # Generate corrected text
        outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        corrected_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the corrected chunk
        corrected_text += corrected_chunk + " "

        # Move to the next chunk
        i += max_length

    return corrected_text.strip()

corrected_sentence = correct_grammar(transcription)

# Function to normalize text by removing punctuation and converting to lowercase
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

corrected_sentence = normalize_text(corrected_sentence)
transcription = normalize_text(transcription)

def grammatical_score(original, corrected):
    original_normalized = normalize_text(original)
    corrected_normalized = normalize_text(corrected)

    diff = difflib.ndiff(original_normalized.split(), corrected_normalized.split())
    changes = list(diff)

    corrected_words = [word for word in changes if word.startswith('+ ')]
    removed_words = [word for word in changes if word.startswith('- ')]

    num_changes = len(corrected_words) + len(removed_words)
    total_words = len(original_normalized.split())

    error_ratio = num_changes / total_words if total_words > 0 else 0

    if error_ratio == 0:
        return 10
    elif error_ratio < 0.05:
        return 9
    elif error_ratio < 0.1:
        return 8
    elif error_ratio < 0.2:
        return 6
    else:
        return 4
grammar_score = grammatical_score(transcription, corrected_sentence)

def get_grammar_feedback(score):
    if score == 10:
        return "Perfect grammar! Keep up the great work! 🎯"
    elif score >= 9:
        return "Very good grammar! A little improvement could make it perfect. 👍"
    elif score >= 8:
        return "Good grammar! There are a few minor mistakes. Keep improving. 😊"
    elif score >= 6:
        return "Grammar needs improvement. Review some rules and practice more. 📝"
    else:
        return "Poor grammar. It might help to practice more and focus on sentence structure. 🚀"

def calculate_speech_pace(audio_file):
    # Load the audio file using pydub to get the duration
    audio_segment = AudioSegment.from_file(audio_file)
    duration = len(audio_segment) / 1000  # pydub returns duration in milliseconds

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service")
            return None
        

    word_count = len(transcription.split())
    minutes = duration / 60
    pace = word_count / minutes if minutes > 0 else 0

    if 130 <= pace <= 160:
        pace_feedback = "Your pace is perfect."
    elif 100 <= pace < 130:
        pace_feedback = "You need to speed up a little bit."
    elif pace < 100:
        pace_feedback = "You are going very slow."
    elif 160 < pace <= 190:
        pace_feedback = "You need to slow down a little bit."
    else:
        pace_feedback = "You are going very fast."

    return pace, transcription, pace_feedback

# Calculate speech pace and get feedback
pace, transcription, pace_feedback = calculate_speech_pace(audio_file_path)
if transcription:
    print("Transcription:", transcription)
    print(f"Your grammatical score was: {grammatical_score(transcription, corrected_sentence)}/10")
    print("Speech Pace (WPM):", pace)
    print("Feedback:", pace_feedback)

# Load Wav2Vec2 model and processor for filler word detection
num_labels = 6
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model.eval()

# Load and resample the audio file for filler word detection
waveform, sample_rate = torchaudio.load(audio_file_path)
waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

# Define chunking parameters
segment_duration = 2
sample_rate = 16000
samples_per_segment = segment_duration * sample_rate

total_duration = waveform.shape[1] / sample_rate
total_segments = total_duration // segment_duration

filler_counts = {"Uh": 0, "Um": 0}
label_map = {0: "Uh", 1: "Words", 2: "Laughter", 3: "Um", 4: "Music", 5: "Breath"}

# Process audio in 2-second chunks for filler word detection
for start in range(0, waveform.shape[1], samples_per_segment):
    end = start + samples_per_segment
    segment = waveform[:, start:end]

    if segment.shape[1] < samples_per_segment:
        break

    inputs = processor(segment.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

    for pred in predictions:
        label = label_map[pred.item()]
        if label in filler_counts:
            filler_counts[label] += 1

# Compute filler rate
total_fillers = filler_counts["Uh"] + filler_counts["Um"]
filler_rate = total_fillers / total_segments if total_segments > 0 else 0

filler_score = max(100 - (filler_rate * 100), 0)

filler_feedback = ""
if filler_score > 90:
    filler_feedback = "Excellent! You used very few filler words. Keep it up! 🎯"
elif filler_score > 75:
    filler_feedback = "Good job! You have some fillers, but it's not distracting. Try to be more mindful. 😊"
elif filler_score > 50:
    filler_feedback = "You're using filler words often. Consider pausing instead of saying 'Uh' or 'Um'. 🧐"
else:
    filler_feedback = "High use of filler words detected! Try slowing down and structuring your thoughts before speaking. 🚀"


# Print results
print(f"\n🎤 Speech Fluency Score: {filler_score:.2f}/100")
print(f"📝 Feedback: {filler_feedback}")
print(f"📊 Filler Word Breakdown: {filler_counts}")

# Initialize the pipeline for pronunciation evaluation
pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Load the audio file (use audio_file_path, not audio_file)
waveform, sample_rate = torchaudio.load(audio_file_path, format="wav")

# Convert stereo to mono if needed
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample to 16kHz if needed (Wav2Vec2 expects 16kHz audio)
if sample_rate != 16000:
    resample = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resample(waveform)

# Normalize the waveform
waveform = (waveform - waveform.mean()) / waveform.std()

# Convert tensor to numpy array (if needed for the pipeline)
waveform = waveform.squeeze(0).numpy()

# Run the pronunciation evaluation model
result = pipe(waveform)

# Define score mapping
score_mapping = {
    "advanced": 100,
    "proficient": 75,
    "intermediate": 50,
    "beginer": 25  # Note: 'beginer' is misspelled in the model output, so keeping it as is
}

# Compute final score based on the model output
final_pronunciation_score = sum(score_mapping[item["label"]] * item["score"] for item in result)

def pronunciation_feedback(score):
    if score >= 90:
        return "Excellent pronunciation! Your articulation is clear and precise. 🎯"
    elif score >= 75:
        return "Good pronunciation. A few improvements can make your speech clearer. 😊"
    elif score >= 50:
        return "Pronunciation could be better. Focus on articulation and clarity. 🧐"
    else:
        return "Needs improvement in pronunciation. Practice speaking more clearly. 🚀"
    
pronunciation_feedback = pronunciation_feedback(final_pronunciation_score)

feedback_report = {
    "Grammar Score": f"{grammar_score}/10",
    "Grammar Feedback": get_grammar_feedback(grammar_score),
    "Speech Pace": f"{pace} WPM",
    "Speech Pace Feedback": pace_feedback,
    "Fluency Score": f"{filler_score}/100",
    "Fluency Feedback": filler_feedback,
    "Filler Word Breakdown": filler_counts,
    "Pronunciation Score": f"{final_pronunciation_score}/100",
    "Pronunciation Feedback": pronunciation_feedback
 
}

# Print the feedback report
print("\nFeedback Report:")
for key, value in feedback_report.items():
    print(f"{key}: {value}")
