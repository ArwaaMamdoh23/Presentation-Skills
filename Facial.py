import cv2
import os
import numpy as np
from collections import Counter
from deepface import DeepFace

def refine_emotion_prediction(emotion, eye_contact):
    
    
    # Emotion and Eye Contact compatibility based on psychological studies
    
    if emotion == "neutral":
        # Neutral emotions might have less distinct eye contact, both with or without eye contact is possible
        if eye_contact == "Eye Contact":
            return "attentive"
        else:
            return "indifferent"  # No strong emotion, could show less engagement
    
    if emotion == "angry":
        # Angry emotions are often associated with direct eye contact
        if eye_contact == "Eye Contact":
            return "intense"  # Direct eye contact with anger intensifies the emotion
        else:
            return "defensive"  # Avoiding eye contact in anger could show defensiveness
    
    if emotion == "fear":
        # Fear is typically linked to avoiding eye contact (avoidance behavior)
        if eye_contact == "Eye Contact":
            return "nervous"  # Fearful but maintaining eye contact can show nervousness or anxiety
        else:
            return "distrust"  # Fear often leads to avoiding eye contact, showing discomfort or distrust
    
    if emotion == "happy":
        # Happiness is often associated with more relaxed and positive body language, which can include eye contact
        if eye_contact == "Eye Contact":
            return "joyful"  # Happiness with eye contact indicates engagement and connection
        else:
            return "content"  # Happiness without eye contact may indicate inner contentment
    
    if emotion == "sad":
        # Sadness is typically associated with avoiding eye contact or looking downward
        if eye_contact == "Eye Contact":
            return "vulnerable"  # Sadness with eye contact can be perceived as vulnerability or needing support
        else:
            return "isolated"  # Avoiding eye contact with sadness shows emotional withdrawal
    
    if emotion == "surprise":
        # Surprise can result in a momentary loss of eye contact or wide-open eyes (increased alertness)
        if eye_contact == "Eye Contact":
            return "alert"  # Surprise with eye contact shows high alertness and focus
        else:
            return "disoriented"  # Surprise without eye contact can indicate disorientation or shock

    # Default case (if the emotion doesn't match any of the above)
    return emotion


def detect_eye_contact(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) > 0:
        return "Eye Contact"
    else:
        return "No Eye Contact"

def preprocess_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = image[y:y + h, x:x + w]
        return image
    return image

def predict_emotions_on_frames(frame):
    processed_image = preprocess_frame(frame)
    analysis = DeepFace.analyze(img_path=processed_image, actions=['emotion'], enforce_detection=False)
    emotion = analysis[0]['dominant_emotion']
    
    eye_contact = detect_eye_contact(processed_image)
    refined_emotion = refine_emotion_prediction(emotion, eye_contact)
    return refined_emotion, eye_contact

video_path = r"D:\GitHub\Presentation-Skills\Videos\happyvid.mp4"
cap = cv2.VideoCapture(video_path)

all_emotions = []
all_eye_contacts = []

is_paused = False
playback_speed = 1

while cap.isOpened():
    if not is_paused:
        ret, frame = cap.read()
    else:
        ret = True 
    
    if not ret:
        break

    emotion, eye_contact = predict_emotions_on_frames(frame)
    all_emotions.append(emotion)
    all_eye_contacts.append(eye_contact)

    frame_resized = cv2.resize(frame, (640, 360))

    cv2.putText(frame_resized, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame_resized, f"Eye Contact: {eye_contact}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Emotion Detection', frame_resized)

    print(f"Frame: Emotion = {emotion}, Eye Contact = {eye_contact}")

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

    if key == ord('p'):
        is_paused = not is_paused  

    if key == ord('+'):
        playback_speed = min(10, playback_speed + 1)  

    if key == ord('-'):
        playback_speed = max(1, playback_speed - 1)

    if is_paused:
        continue  
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + playback_speed - 1)  

if all_emotions:
    dominant_emotion = Counter(all_emotions).most_common(1)[0][0]  
    print(f"Dominant Emotion: {dominant_emotion}")

if all_eye_contacts:
    dominant_eye_contact = Counter(all_eye_contacts).most_common(1)[0][0]  
    print(f"Dominant Eye Contact: {dominant_eye_contact}")

cap.release()
cv2.destroyAllWindows()
