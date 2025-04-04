import speech_recognition as sr
from pydub import AudioSegment
from transformers import T5ForConditionalGeneration, T5Tokenizer
import difflib
import numpy as np
import spacy 
import string
import aifc

recognizer = sr.Recognizer()


audio_file_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/presentation-test.wav"

audio_segment = AudioSegment.from_file(audio_file_path)

audio_duration = len(audio_segment) / 1000  
audio_file = sr.AudioFile(audio_file_path)

with audio_file as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)  
    chunk_size = 30

    transcription = ""  
    current_position = 0

    while current_position < audio_duration:
        try:
            audio_chunk = recognizer.record(source, duration=chunk_size)

            chunk_text = recognizer.recognize_google(audio_chunk)
            transcription += chunk_text + " "  
            print(f"Transcribed chunk {current_position}-{current_position+chunk_size}s: {chunk_text}")
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand the audio between {current_position}s and {current_position+chunk_size}s")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

        current_position += chunk_size


model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

def correct_grammar(sentence, max_length=512):
    corrected_text = ""
    i = 0
    while i < len(sentence):
        chunk = sentence[i:i + max_length]
        input_text = "grammar: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

        outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        corrected_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)

        corrected_text += corrected_chunk + " "

        i += max_length

    return corrected_text.strip()


corrected_sentence = correct_grammar(transcription)

def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

corrected_sentence= normalize_text(corrected_sentence)
transcription=normalize_text(transcription)


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


def calculate_speech_pace(audio_file):
    audio_segment = AudioSegment.from_file(audio_file)
    
    duration = len(audio_segment) / 1000  
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
    else:  # pace > 190
        pace_feedback = "You are going very fast."

    return pace, transcription, pace_feedback


audio_file = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/presentation-test.wav"  
pace, transcription, pace_feedback = calculate_speech_pace(audio_file)
if transcription:
    print("Transcription:", transcription)
    print(f"Your grammatical score was: {grammatical_score(transcription,corrected_sentence)}/10")
    print("Speech Pace (WPM):", pace)
    print("Feedback:", pace_feedback)


