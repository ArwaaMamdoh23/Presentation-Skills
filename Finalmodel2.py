import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import mediapipe as mp
from deepface import DeepFace
from collections import Counter
import torchaudio
import torchaudio.transforms as T
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, pipeline, T5ForConditionalGeneration, T5Tokenizer
import ffmpeg
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import difflib
import string
import spacy
import aifc
import os
from moviepy import VideoFileClip

# Function to load models
# Initialize the model variables globally
posenet_model = None
mp_hands = None
mp_drawing = None
wav2vec2_model = None
processor = None
pronunciation_pipe = None
t5_model = None
tokenizer = None

def load_models():
    global posenet_model, mp_hands, wav2vec2_model, processor, pronunciation_pipe, t5_model, tokenizer
    
    # Load PoseNet model for posture detection (PostureNet)
    posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    try:
        posenet_model = hub.load(posenet_model_url)
        print("PoseNet model loaded successfully")
    except Exception as e:
        print(f"Error loading PoseNet model: {e}")

    # MediaPipe Hands for gesture recognition
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe Hands model initialized")

    # Load Wav2Vec2 model for sequence classification
    num_labels = 6
    wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec2_model.eval()
    print("Wav2Vec2 model loaded successfully")

    # Pipeline for pronunciation evaluation
    pronunciation_pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")
    print("Pronunciation evaluation pipeline initialized")

    # Load the pre-trained T5 model for grammar correction
    t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    print("T5 model for grammar correction loaded successfully")

# Call the load_models function when the script is run directly
if __name__ == "__main__":
    load_models()
