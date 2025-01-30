import speech_recognition as sr
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import difflib
import numpy as np
import spacy 
import string
import aifc

# Initialize recognizer class (for recognizing speech)
recognizer = sr.Recognizer()

# Path to your audio file (change the path as needed)
# audio_file_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/WhatsApp Audio 2024-10-21 at 8.55.53 PM.wav"
audio_file_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/presentation-test.wav"

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

# Print the full transcription
# print(f"Full Transcription:\n{transcription}")

# Load the pre-trained T5 model for grammar correction
model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

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

# Example usage

corrected_sentence = correct_grammar(transcription)

# Function to normalize text by removing punctuation and converting to lowercase
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

corrected_sentence= normalize_text(corrected_sentence)
transcription=normalize_text(transcription)


def grammatical_score(original, corrected):
    # Normalize both the original and corrected sentences
    original_normalized = normalize_text(original)
    corrected_normalized = normalize_text(corrected)

    # Using difflib to compare the normalized sentences
    diff = difflib.ndiff(original_normalized.split(), corrected_normalized.split())
    changes = list(diff)

    # Count the number of words that changed (added/removed/modified)
    corrected_words = [word for word in changes if word.startswith('+ ')]
    removed_words = [word for word in changes if word.startswith('- ')]

    num_changes = len(corrected_words) + len(removed_words)
    total_words = len(original_normalized.split())

    # Calculate the error ratio (number of corrections divided by total words in the original text)
    error_ratio = num_changes / total_words if total_words > 0 else 0

    # Score based on the error ratio, normalizing it with respect to the length
    if error_ratio == 0:
        return 10  # Perfect grammar
    elif error_ratio < 0.05:
        return 9  # Very few errors (minor grammar issues)
    elif error_ratio < 0.1:
        return 8  # A few grammar issues
    elif error_ratio < 0.2:
        return 6  # Noticeable grammar issues
    else:
        return 4  # Many grammatical errors

# print(f"Your grammatical score was: {grammatical_score(transcription,corrected_sentence)}/10")

def calculate_speech_pace(audio_file):
    # Load the audio file using pydub to get the duration
    audio_segment = AudioSegment.from_file(audio_file)
    
    # Get the duration of the audio in seconds
    duration = len(audio_segment) / 1000  # pydub returns duration in milliseconds

    # Transcribe audio to text using SpeechRecognition
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

    # Count the number of words in the transcription
    word_count = len(transcription.split())
    # Calculate pace in words per minute (WPM)
    minutes = duration / 60
    pace = word_count / minutes if minutes > 0 else 0
    # Determine pace category
    if 130 <= pace <= 160:
        pace_feedback = "Your pace is perfect."
    elif 100 <= pace < 130:
        pace_feedback = "You need to speed up a little bit."
    elif pace < 100:
        pace_feedback = "You are going very slow."
    elif 160 < pace <= 190:
        pace_feedback = "You need to slow down a little bit."
    else:  # pace > 190
        pace_feedback = "You are going very fast."

    return pace, transcription, pace_feedback

# Example usage

# audio_file = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/WhatsApp Audio 2024-10-21 at 8.55.53 PM.wav"  # Specify your audio file path here
audio_file = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/presentation-test.wav"  # Specify your audio file path here
pace, transcription, pace_feedback = calculate_speech_pace(audio_file)
if transcription:
    print("Transcription:", transcription)
    print(f"Your grammatical score was: {grammatical_score(transcription,corrected_sentence)}/10")
    print("Speech Pace (WPM):", pace)
    print("Feedback:", pace_feedback)


