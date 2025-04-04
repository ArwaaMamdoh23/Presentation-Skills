import os
import numpy as np
import pickle
import librosa
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

model_path = r"speech emotion/CNN_Model_Weights (1).h5"
loaded_model = load_model(model_path)
loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

scaler_path = r"speech emotion/D__Speech Emotion Recognition-20250207T115414Z-001_Speech Emotion Recognition_Scaler.pickle"
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

EXPECTED_FEATURES = 2376  

def compute_zcr(audio_data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length, hop_length=hop_length))

def compute_rmse(audio_data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length))

def compute_mfcc(audio_data, sample_rate=22050, n_mfcc=13):
    return np.ravel(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc))

def extract_audio_features(audio_data, sample_rate=22050):
    """Extracts features and ensures fixed-length input."""
    zcr = compute_zcr(audio_data)
    rmse = compute_rmse(audio_data)
    mfcc = compute_mfcc(audio_data, sample_rate)

    features = np.hstack((zcr, rmse, mfcc))

    if len(features) < EXPECTED_FEATURES:
        features = np.pad(features, (0, EXPECTED_FEATURES - len(features)), mode="constant")
    else:
        features = features[:EXPECTED_FEATURES]

    return features

def get_prediction_features(file_path):
    """Extracts and processes features for emotion prediction."""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=22050)
    except:
        return None

    features = extract_audio_features(audio_data, sample_rate)
    reshaped_features = np.reshape(features, (1, EXPECTED_FEATURES))
    scaled_features = scaler.transform(reshaped_features)

    return np.expand_dims(scaled_features, axis=2)

def predict_emotion(file_path):
    """Predicts emotion from an audio file."""
    emotion_mapping = {
        1: 'Neutral', 2: 'Fear', 3: 'Happy', 4: 'Sad',
        5: 'Angry', 6: 'Disgust', 7: 'Surprise'
    }

    processed_features = get_prediction_features(file_path)
    if processed_features is None:
        return None

    prediction_scores = loaded_model.predict(processed_features)
    predicted_label = np.argmax(prediction_scores, axis=1)[0]
    return emotion_mapping.get(predicted_label, 'Unknown')

def extract_audio_chunks(video_path, output_folder, chunk_length=3000):
    os.makedirs(output_folder, exist_ok=True)  
    video = VideoFileClip(video_path)
    audio_path = os.path.join(output_folder, "full_audio.wav")
    video.audio.write_audiofile(audio_path)  

    audio = AudioSegment.from_wav(audio_path)
    total_duration = len(audio)  

    chunk_files = []  
    for i in range(0, total_duration, chunk_length):
        chunk = audio[i:i+chunk_length]  
        chunk_file = os.path.join(output_folder, f"chunk_{i//1000}-{(i+chunk_length)//1000}.wav")
        chunk.export(chunk_file, format="wav")
        chunk_files.append(chunk_file)

    return chunk_files  

def get_dominant_emotion(video_path):
    audio_folder = "audio_chunks"  
    audio_chunks = extract_audio_chunks(video_path, audio_folder)  

    emotion_counts = {}

    for chunk in audio_chunks:
        emotion = predict_emotion(chunk)  
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1  

    return max(emotion_counts, key=emotion_counts.get) if emotion_counts else None

if __name__ == "__main__":
    video_path = r"C:\Users\MOUSTAFA\Videos\Captures\HAPPY In Cinema - YouTube - Google Chrome 2025-02-18 15-25-12.mp4"
    dominant_emotion = get_dominant_emotion(video_path)
    
    if dominant_emotion:
        print(f"The emotion recognized from your video is: {dominant_emotion}")
    else:
        print("No  emotion could be recognized from the video.")
