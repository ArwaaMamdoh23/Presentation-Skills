from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from moviepy import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import logging
from collections import Counter
from deepface import DeepFace
import torchaudio
import torchaudio.transforms as T
import tensorflow_hub as hub
import mediapipe as mp
import difflib
import speech_recognition as sr
from pydub import AudioSegment
import tensorflow as tf
import requests
from argostranslate import package, translate
import whisper
import librosa
import string
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2Processor, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    pipeline
)
from supabase import create_client, Client
import boto3
from botocore.exceptions import NoCredentialsError
import uuid
from datetime import datetime
from flask_cors import CORS




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv('SUPABASE_URL', 'https://ohllbliwedftnyqmthze.supabase.co'),
    os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obGxibGl3ZWRmdG55cW10aHplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM1NDIxMzAsImV4cCI6MjA1OTExODEzMH0.XW1XNf7v3-JX94-1xJNgPM70t2qvZoEClyAab85ie1o')
)

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')  # Add your AWS region
)

# Add AWS configuration
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME', 'your-bucket-name')
AWS_CLOUDFRONT_DOMAIN = os.getenv('AWS_CLOUDFRONT_DOMAIN', 'your-cloudfront-domain')

# Initialize models and configurations
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load Wav2Vec2 model for filler word detection
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=6)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model.eval()

# Load T5 model for grammar correction
t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

# Load PoseNet Model for posture detection
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
posenet_model = hub.load(posenet_model_url)
movenet = posenet_model.signatures['serving_default']

# Load the pronunciation evaluation pipeline
pronunciation_pipe = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Load Whisper model
whisper_model = whisper.load_model("base")

# Define mappings and counters
gesture_to_body_language = {
    "Open Palm": "Honesty",
    "Closed Fist": "Determination",
    "Pointing Finger": "Confidence",
    "Thumbs Up": "Encouragement",
    "Thumbs Down": "Criticism",
    "Victory Sign": "Peace",
    "OK Sign": "Agreement",
    "Rock Sign": "Excitement",
    "Call Me": "Friendly"
}

posture_meanings = {
    "Head Up": "Confidence",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Arms on Hips": "Confidence",
    "Crossed Arms": "Defensive",
    "Hands in Pockets": "Casual"
}

# Initialize counters
posture_counter = {posture: 0 for posture in posture_meanings.keys()}
gesture_counter = {gesture: 0 for gesture in gesture_to_body_language.keys()}

def extract_audio_from_video(video_file_path):
    """Extract audio from video file and save as WAV."""
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]
    audio_file_path = os.path.join(os.path.dirname(video_file_path), f"{base_name}.wav")
    video = VideoFileClip(video_file_path)
    audio = video.audio
    audio.write_audiofile(audio_file_path)
    return audio_file_path

def predict_emotions(image):
    """Predict emotions from image using DeepFace."""
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return "Unknown"

def detect_eye_contact(image):
    """Detect eye contact in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    return "Eye Contact" if len(eyes) > 0 else "No Eye Contact"

def refine_emotion_prediction(emotion, eye_contact):
    """Refine emotion prediction based on eye contact."""
    if emotion == "neutral":
        return "attentive" if eye_contact == "Eye Contact" else "indifferent"
    if emotion == "angry":
        return "intense" if eye_contact == "Eye Contact" else "defensive"
    if emotion == "fear":
        return "nervous" if eye_contact == "Eye Contact" else "distrust"
    if emotion == "happy":
        return "joyful" if eye_contact == "Eye Contact" else "content"
    if emotion == "sad":
        return "vulnerable" if eye_contact == "Eye Contact" else "isolated"
    if emotion == "surprise":
        return "alert" if eye_contact == "Eye Contact" else "disoriented"
    return emotion

def classify_hand_gesture(hand_landmarks):
    """Classify hand gestures based on landmark positions."""
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19

    def is_finger_extended(tip, dip):
        return landmarks[tip][1] < landmarks[dip][1]

    thumb_extended = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]
    index_extended = is_finger_extended(INDEX_TIP, INDEX_DIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_DIP)
    ring_extended = is_finger_extended(RING_TIP, RING_DIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_DIP)

    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm"
    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist"
    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger"
    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up"
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down"
    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign"
    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign"
    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign"
    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me"
    return "Unknown Gesture"

def get_posture_feedback(posture):
    """Get feedback based on detected posture."""
    feedback = {
        "Head Up": "Good posture! Keep your head up to show confidence.",
        "Slouching": "Try to stand up straight. Slouching can make you appear less confident.",
        "Leaning Forward": "Your leaning forward shows interest, but be careful not to invade personal space.",
        "Head Down": "Try to keep your head up to maintain eye contact and show confidence.",
        "Leaning Back": "Leaning back can appear relaxed, but make sure you're still engaged with your audience.",
        "Arms on Hips": "This posture shows confidence, but be mindful of appearing too aggressive.",
        "Crossed Arms": "Crossed arms can appear defensive. Try to keep your arms open to appear more approachable.",
        "Hands in Pockets": "Hands in pockets can appear casual, but might make you look less professional.",
        "Unknown Posture": "Try to maintain a balanced and confident posture."
    }
    return feedback.get(posture, "Maintain a balanced and confident posture.")

def get_gesture_feedback(gesture):
    """Get feedback based on detected gesture."""
    feedback = {
        "Open Palm": "Your open palm gestures show honesty and openness.",
        "Closed Fist": "Your closed fist shows determination, but be careful not to appear aggressive.",
        "Pointing Finger": "Pointing can be effective for emphasis, but use it sparingly to avoid appearing accusatory.",
        "Thumbs Up": "Thumbs up shows encouragement and positivity.",
        "Thumbs Down": "Be careful with thumbs down as it can appear negative.",
        "Victory Sign": "Victory sign shows enthusiasm, but use it appropriately for the context.",
        "OK Sign": "OK sign shows agreement, but be aware of cultural differences in interpretation.",
        "Rock Sign": "Rock sign shows excitement, but make sure it's appropriate for your audience.",
        "Call Me": "Call me gesture can be friendly, but ensure it's appropriate for the setting.",
        "Unknown Gesture": "Try to use clear and purposeful hand gestures to enhance your message."
    }
    return feedback.get(gesture, "Use clear and purposeful hand gestures to enhance your message.")

def get_emotion_feedback(refined_emotion, eye_contact):
    """Get feedback based on emotion and eye contact."""
    feedback = {
        "attentive": "You appear attentive. Keep your facial expressions engaging.",
        "indifferent": "Try to add more warmth to your expression for a more approachable look.",
        "intense": "Your intensity shows determination, but be mindful not to appear hostile.",
        "defensive": "Lack of eye contact with anger may appear defensive. Try to calm down and engage more openly.",
        "nervous": "Nervousness is noticeable. Maintain steady eye contact to show confidence.",
        "distrust": "Avoid avoiding eye contact; it can make you appear untrustworthy. Try to relax and engage more.",
        "joyful": "You're radiating joy! Keep the positivity and maintain eye contact for a more connected look.",
        "content": "You're happy, but make sure to engage with your audience by maintaining eye contact.",
        "vulnerable": "You seem vulnerable. Try smiling to lighten the mood if you're comfortable.",
        "isolated": "Lack of eye contact combined with sadness may appear disengaged. Try to make eye contact for a stronger presence.",
        "alert": "You seem alert! Maintain eye contact to help convey your surprise more clearly.",
        "disoriented": "You're surprised but seem disconnected. Try focusing and engaging with your audience."
    }
    return refined_emotion, feedback.get(refined_emotion, "Try to maintain a balanced expression to convey clarity.")

def get_grammar_feedback(score):
    """Get feedback based on grammar score."""
    if score >= 90:
        return "Excellent grammar! Your speech is very well-structured."
    elif score >= 80:
        return "Good grammar! There are minor improvements that could be made."
    elif score >= 70:
        return "Fair grammar. Consider reviewing your sentence structure."
    else:
        return "Your grammar needs improvement. Consider practicing more complex sentence structures."

def pronunciation_feedback(score):
    """Get feedback based on pronunciation score."""
    if score >= 90:
        return "Excellent pronunciation! Your speech is very clear."
    elif score >= 80:
        return "Good pronunciation! There are minor improvements that could be made."
    elif score >= 70:
        return "Fair pronunciation. Consider practicing difficult words more."
    else:
        return "Your pronunciation needs improvement. Consider practicing with a pronunciation guide."

def detect_interruption(audio_chunk, previous_speech_segment, silence_threshold=0.5, pace_change_threshold=50):
    """Detect interruptions in speech based on silence and pace changes."""
    silence = np.mean(np.abs(audio_chunk)) < silence_threshold
    pace_change = abs(len(audio_chunk) - len(previous_speech_segment)) > pace_change_threshold
    return silence or pace_change

def analyze_post_interruption_speech(posture, gesture, emotion, eye_contact, interruptions):
    """Analyze speaker's response to interruptions and provide feedback."""
    feedback = set()  # Using a set to store feedback and eliminate duplicates
    audience_feedback = []  # Will hold the audience interaction feedback specifically

    # Check if the interruptions list is empty
    if not interruptions:
        # No interruptions, add the appropriate feedback for no interaction
        audience_feedback.append("No interaction with the audience detected.")
    else:
        # If there were interruptions, process feedback
        for interruption in interruptions:
            # Analyzing posture after interruption
            if posture == "Crossed Arms":
                feedback.add("The speaker seems defensive after the interruption.")
            elif posture == "Leaning Back":
                feedback.add("The speaker seems relaxed after the interruption.")
            elif posture == "Arms on Hips":
                feedback.add("The speaker appears confident but might be aggressive after the interruption.")
            elif posture == "Slouching":
                feedback.add("The speaker seems disengaged or insecure after the interruption.")
            elif posture == "Head Down":
                feedback.add("The speaker may feel defeated or unsure after the interruption.")
            elif posture == "Head Up":
                feedback.add("The speaker seems confident and unphased after the interruption.")
            elif posture == "Hands in Pockets":
                feedback.add("The speaker seems distant or detached after the interruption.")
            else:
                feedback.add("Unrecognized posture. Please check the input.")

            # Analyzing gestures after interruption
            if gesture == "Thumbs Up":
                feedback.add("The speaker is reassuring and positive despite the interruption.")
            elif gesture == "Thumbs Down":
                feedback.add("The speaker is likely displeased or frustrated after the interruption.")
            elif gesture == "OK Sign":
                feedback.add("The speaker may be trying to convey agreement but seems hesitant.")
            elif gesture == "Victory Sign":
                feedback.add("The speaker shows confidence and success, but might be mocking after the interruption.")
            elif gesture == "Closed Fist":
                feedback.add("The speaker seems determined but possibly frustrated after the interruption.")
            elif gesture == "Pointing Finger":
                feedback.add("The speaker may be emphasizing a point more forcefully after the interruption.")
            elif gesture == "Open Palm":
                feedback.add("The speaker is showing honesty and openness after the interruption.")
            else:
                feedback.add("Unrecognized gesture. Please check the input.")
            
            # Analyzing emotion after interruption
            if emotion == "angry":
                feedback.add("The speaker seems angry after the interruption.")
            elif emotion == "happy":
                feedback.add("The speaker seems joyful and unaffected by the interruption.")
            elif emotion == "sad":
                feedback.add("The speaker may feel discouraged or upset after the interruption.")
            elif emotion == "fear":
                feedback.add("The speaker seems nervous or anxious after the interruption.")
            elif emotion == "surprise":
                feedback.add("The speaker is taken aback by the interruption and looks surprised.")
            elif emotion == "neutral" or emotion not in ["angry", "happy", "sad", "fear", "surprise"]:
                feedback.add("The speaker maintains a neutral expression, unaffected by the interruption.")
            elif emotion == "attentive":
                feedback.add("You appear attentive. Keep your facial expressions engaging.")
            else:
                feedback.add("Unrecognized emotion. Please check the input.")
            
            # Analyzing eye contact after interruption
            if eye_contact == "No Eye Contact":
                feedback.add("The speaker avoids eye contact after the interruption, possibly indicating discomfort.")
            elif eye_contact == "Eye Contact":
                feedback.add("The speaker maintains eye contact, showing confidence despite the interruption.")
            else:
                feedback.add("Unrecognized eye contact. Please check the input.")

    # Combine all feedback into the final report
    if audience_feedback:
        return audience_feedback
    else:
        return list(feedback)

def run_inference(frame):
    """Run PoseNet inference on the frame."""
    # Resize and pad the image to keep aspect ratio
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    
    # Run inference
    results = movenet(img)
    return results

def extract_keypoints(results):
    """Extract keypoints from PoseNet results."""
    keypoints = results['output_0'].numpy()[0][0]
    return keypoints

def classify_posture(keypoints):
    """Classify posture based on keypoint positions."""
    # Extract relevant keypoints
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    # Calculate angles and positions
    shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], 
                              right_shoulder[0] - left_shoulder[0])
    hip_angle = np.arctan2(right_hip[1] - left_hip[1], 
                          right_hip[0] - left_hip[0])
    
    # Calculate vertical alignment
    shoulder_hip_diff = abs((left_shoulder[1] + right_shoulder[1])/2 - 
                          (left_hip[1] + right_hip[1])/2)
    
    # Head position relative to shoulders
    head_position = nose[1] - (left_shoulder[1] + right_shoulder[1])/2
    
    # Classify posture based on angles and positions
    if abs(shoulder_angle) > 0.3:  # Slouching
        return "Slouching"
    elif head_position < -0.1:  # Head down
        return "Head Down"
    elif head_position > 0.1:  # Head up
        return "Head Up"
    elif shoulder_hip_diff > 0.2:  # Leaning
        if nose[0] < (left_shoulder[0] + right_shoulder[0])/2:
            return "Leaning Back"
        else:
            return "Leaning Forward"
    else:
        return "Neutral"

def correct_grammar(text, model, tokenizer):
    """Correct grammar in the given text using T5 model."""
    # Prepare input text
    input_text = f"grammar: {text}"
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate correction
    outputs = model.generate(
        input_ids,
        max_length=512,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode and return corrected text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def grammatical_score(original_text, corrected_text):
    """Calculate grammatical correctness score based on differences between original and corrected text."""
    # Convert texts to lowercase for comparison
    original = original_text.lower()
    corrected = corrected_text.lower()
    
    # Calculate similarity using difflib
    matcher = difflib.SequenceMatcher(None, original, corrected)
    similarity_ratio = matcher.ratio()
    
    # Convert similarity ratio to a score out of 100
    score = int(similarity_ratio * 100)
    
    return score

def upload_to_s3(file_path, bucket_name):
    """Upload a file to S3 bucket."""
    try:
        file_name = f"{uuid.uuid4()}_{os.path.basename(file_path)}"
        s3_client.upload_file(file_path, bucket_name, file_name)
        return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return None

def save_feedback_to_supabase(feedback_data, user_id, video_url):
    """Save feedback data to Supabase."""
    try:
        data = {
            'user_id': user_id,
            'video_url': video_url,
            'feedback_data': feedback_data,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'completed'
        }
        
        result = supabase.table('presentation_feedback').insert(data).execute()
        return result.data[0]['id']
    except Exception as e:
        logger.error(f"Error saving to Supabase: {str(e)}")
        return None

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and processing."""
    try:
        # Get user_id from request
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400
        
        # Get video file from request
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
            
        video_file = request.files['video']
        if not video_file:
            return jsonify({"error": "Empty video file"}), 400

        # Create uploads directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)

        # Save video temporarily
        video_path = os.path.join('uploads', f"{uuid.uuid4()}.mp4")
        video_file.save(video_path)

        # Extract audio from the video
        audio_file_path = extract_audio_from_video(video_path)
        logger.info(f"Audio extracted to: {audio_file_path}")

        # Process the video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        all_emotions = []
        all_eye_contacts = []
        all_gestures = []
        all_postures = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 10 == 0:  # Process every 10th frame
                # Emotion and eye contact detection
                emotion = predict_emotions(frame)
                eye_contact = detect_eye_contact(frame)
                refined_emotion = refine_emotion_prediction(emotion, eye_contact)
                
                # Gesture detection
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_hands = mp_hands.process(image_rgb)
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        gesture = classify_hand_gesture(hand_landmarks)
                        if gesture != "Unknown Gesture":
                            all_gestures.append(gesture)
                
                # Posture detection
                results = run_inference(frame)
                keypoints = extract_keypoints(results)
                posture = classify_posture(keypoints)
                if posture != "Unknown Posture":
                    all_postures.append(posture)
                
                all_emotions.append(refined_emotion)
                all_eye_contacts.append(eye_contact)

            frame_count += 1

        cap.release()

        # Get dominant metrics
        dominant_emotion = Counter(all_emotions).most_common(1)[0][0] if all_emotions else "Unknown"
        dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0] if all_eye_contacts else "Unknown"
        dominant_gesture = Counter(all_gestures).most_common(1)[0][0] if all_gestures else "Unknown"
        dominant_posture = Counter(all_postures).most_common(1)[0][0] if all_postures else "Unknown"

        # Process audio for speech analysis
        waveform, sr = librosa.load(audio_file_path, sr=16000)
        waveform = waveform / max(abs(waveform))  # Normalize
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(audio=waveform)
        transcription = result["text"]
        detected_lang = result["language"]

        # Calculate speech pace
        audio_segment = AudioSegment.from_file(audio_file_path)
        duration = len(audio_segment) / 1000  # seconds
        word_count = len(transcription.split())
        minutes = duration / 60
        pace = word_count / minutes if minutes > 0 else 0

        # Generate pace feedback
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

        # Get pronunciation score
        pronunciation_result = pronunciation_pipe(waveform)
        pronunciation_score = max(pronunciation_result[0]['score'] * 100, 0)
        pronunciation_feedback_text = pronunciation_feedback(pronunciation_score)
        
        # Get grammar score
        corrected_text = correct_grammar(transcription, t5_model, tokenizer)
        grammar_score = grammatical_score(transcription, corrected_text)
        grammar_feedback_text = get_grammar_feedback(grammar_score)

        # Process audio for interruption detection
        waveform, sr = librosa.load(audio_file_path, sr=16000)
        waveform = waveform / max(abs(waveform))  # Normalize
        
        # Define chunking parameters
        segment_duration = 2  # in seconds
        samples_per_segment = segment_duration * sr
        
        # Process the audio in chunks
        interruptions = []
        previous_speech_segment = []
        
        for start in range(0, len(waveform), samples_per_segment):
            end = start + samples_per_segment
            segment = waveform[start:end]
            
            if len(segment) < samples_per_segment:
                break
                
            interruption_detected = detect_interruption(segment, previous_speech_segment)
            if interruption_detected:
                interruptions.append((start, end))
                
            previous_speech_segment = segment

        # Get feedback for each aspect
        _, emotion_feedback_text = get_emotion_feedback(dominant_emotion, dominant_eye_contact)
        gesture_feedback_text = get_gesture_feedback(dominant_gesture)
        posture_feedback_text = get_posture_feedback(dominant_posture)

        # Analyze audience interaction
        audience_interaction = analyze_post_interruption_speech(
            posture=dominant_posture,
            gesture=dominant_gesture,
            emotion=dominant_emotion,
            eye_contact=dominant_eye_contact,
            interruptions=interruptions
        )

        # Prepare feedback data
        feedback_data = {
            "emotion_analysis": {
                "dominant_emotion": dominant_emotion,
                "eye_contact": dominant_eye_contact,
                "feedback": emotion_feedback_text
            },
            "gesture_analysis": {
                "dominant_gesture": dominant_gesture,
                "feedback": gesture_feedback_text
            },
            "posture_analysis": {
                "dominant_posture": dominant_posture,
                "feedback": posture_feedback_text
            },
            "speech_analysis": {
                "transcription": transcription,
                "detected_language": detected_lang,
                "pace": pace,
                "pace_feedback": pace_feedback,
                "pronunciation_score": pronunciation_score,
                "pronunciation_feedback": pronunciation_feedback_text,
                "grammar_score": grammar_score,
                "grammar_feedback": grammar_feedback_text,
                "corrected_text": corrected_text
            },
            "audience_interaction": {
                "feedback": audience_interaction,
                "interruptions_detected": len(interruptions)
            }
        }
            
        # Clean up temporary files
        os.remove(video_path)
        os.remove(audio_file_path)
        
        return jsonify(feedback_data)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)