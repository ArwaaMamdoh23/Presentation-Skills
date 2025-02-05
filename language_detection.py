import whisper
import moviepy.editor as mp
import librosa
import numpy as np

# Load the Whisper model
model = whisper.load_model("base")

# Function to load and preprocess audio
def load_audio(file_path, target_sample_rate=16000):
    try:
        # Load the audio file using librosa (don't resample yet)
        audio, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
        
        # Check if resampling is needed and perform it
        if sr != target_sample_rate:
            # Resample audio to match Whisper's expected sample rate (16000)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        
        # Normalize audio to range [-1, 1] for Whisper
        audio = audio / np.max(np.abs(audio))
        return audio
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_output_path="extracted_audio.wav"):
    try:
        # Load the video file using moviepy
        video = mp.VideoFileClip(video_path)
        # Extract audio from the video
        audio = video.audio
        # Write the audio to a WAV file
        audio.write_audiofile(audio_output_path)
        print(f"Audio extracted and saved to {audio_output_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

# Path to the video file
video_path =r"C:\Users\MOUSTAFA\Videos\Captures\Only In Mandarin - Fresh Off The Boat - YouTube - Google Chrome 2025-01-19 21-37-55.mp4"
audio_path = r"C:\Users\MOUSTAFA\Videos\extracted_audio.wav"

# Extract audio from the video
extract_audio_from_video(video_path, audio_path)

# Load and preprocess the extracted audio
audio = load_audio(audio_path)

if audio is not None:
    # Transcribe the audio using Whisper
    result = model.transcribe(audio)
    
    # Print the transcription and detected language
    print("Transcription:", result["text"])
    print("Detected language:", result["language"])
else:
    print("Failed to load audio.")
