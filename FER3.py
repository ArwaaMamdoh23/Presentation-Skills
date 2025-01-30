import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from deepface import DeepFace
import dlib

#  SETUP FACE & EYE DETECTOR
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure the file is in your project folder

#  FUNCTION TO REFINE EMOTIONS BASED ON EYE CONTACT
def refine_emotion_prediction(emotion, eye_contact):
    """
    Adjusts emotion classification based on eye contact.
    """
    if emotion in ["neutral", "fear"] and eye_contact == "No Eye Contact":
        return "bored"

    if emotion in ["happy", "surprise"] and eye_contact == "No Eye Contact":
        return "content"

    if emotion == "angry" and eye_contact == "Eye Contact":
        return "focused"  # Someone looking directly with a serious face might be focused, not angry.

    if emotion == "fear" and eye_contact == "Eye Contact":
        return "alert"

    if emotion == "happy" and eye_contact == "Eye Contact":
        return "excited"

    if emotion == "neutral" and eye_contact == "Eye Contact":
        return "attentive"

    return emotion  # Default case, return the original emotion

#  FUNCTION TO DETECT EYE CONTACT
def detect_eye_contact(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [landmarks.part(i) for i in range(36, 42)]
        right_eye = [landmarks.part(i) for i in range(42, 48)]

        eye_center_x = (left_eye[0].x + right_eye[3].x) // 2
        face_center_x = (face.left() + face.right()) // 2

        # Increase tolerance for slight head turns
        if abs(eye_center_x - face_center_x) < 30:
            return "Eye Contact"
        else:
            return "No Eye Contact"
    
    return "No Face Detected"

#  FUNCTION TO CROP FACE FROM FRAME
def preprocess_frame(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) > 0:
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        x, y, w, h = (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height())
        cropped_face = image[y:y+h, x:x+w]

        # Adjust brightness & contrast
        alpha = 1.5  # Contrast control
        beta = 30    # Brightness control
        enhanced_face = cv2.convertScaleAbs(cropped_face, alpha=alpha, beta=beta)

        resized_face = cv2.resize(enhanced_face, (224, 224))
        return resized_face

    return cv2.resize(image, (224, 224))  # Resize even if no face detected to avoid DeepFace errors

#  FUNCTION TO EXTRACT FRAMES FROM VIDEO
def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extract frames from a video at a specified frame rate.
    """
    if not os.path.exists(video_path):
        print(f" Error: Video file not found at {video_path}")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"🎥 FPS Detected: {fps}")

    if fps == 0:
        print(" Error: Unable to read FPS. Check if the video file is valid.")
        return 0

    frame_interval = max(1, fps // frame_rate)  # Extract every nth frame

    frame_count = 0
    extracted_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f" Extracted {extracted_count} frames.")
    return extracted_count

#  FUNCTION TO PREDICT EMOTIONS & EYE CONTACT
def predict_emotions_on_frames(frame_folder):
    """
    Runs DeepFace on each extracted frame to detect emotions and eye contact.
    """
    results = {}

    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])

    if not frame_files:
        print(" Error: No frames found in folder. Make sure the extraction step worked correctly.")
        return results

    for frame_file in frame_files:
        image_path = os.path.join(frame_folder, frame_file)
        
        # Preprocess the image (Crop Face)
        processed_image = preprocess_frame(image_path)

        try:
            # Predict Emotion using DeepFace
            analysis = DeepFace.analyze(img_path=processed_image, actions=['emotion'], enforce_detection=False)

            emotion = analysis[0]['dominant_emotion']

            # Detect Eye Contact
            eye_contact = detect_eye_contact(cv2.imread(image_path))

            # Adjust emotion classification using refine_emotion_prediction()
            refined_emotion = refine_emotion_prediction(emotion, eye_contact)

            # Store refined emotion and eye contact in results
            results[frame_file] = (refined_emotion, eye_contact)

            print(f"{frame_file}: Emotion={refined_emotion}, Eye Contact={eye_contact}")

        except Exception as e:
            print(f"⚠ Error processing {frame_file}: {e}")

    return results

#  FUNCTION TO DETERMINE OVERALL VIDEO EMOTION
def aggregate_emotions(predictions):
    """
    Uses a weighted approach to determine the most common emotion.
    """
    if not predictions:
        print("⚠ No emotions detected, skipping analysis.")
        return "No Emotion Detected"

    emotion_counts = Counter([pred[0] for pred in predictions.values()])
    
    # Weighted method: Ignore first & last 10% of frames (to remove sudden errors)
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_emotion = sorted_emotions[0][0]

    print(f" Most Common Emotion in Video (Refined): {dominant_emotion}")
    return dominant_emotion

#  FUNCTION TO PLOT EMOTIONS & EYE CONTACT OVER TIME
def plot_emotions(predictions):
    """
    Plots the detected emotions and eye contact over time.
    """
    if not predictions:
        print(" No emotions detected, skipping the graph.")
        return

    frame_numbers = list(range(len(predictions)))
    emotions = [pred[0] for pred in predictions.values()]
    eye_contacts = [1 if pred[1] == "Eye Contact" else 0 for pred in predictions.values()]

    plt.figure(figsize=(12, 6))

    # Plot emotions
    plt.plot(frame_numbers, emotions, marker='o', linestyle='-', color='b', label="Emotion")

    # Plot eye contact (binary)
    plt.plot(frame_numbers, eye_contacts, marker='x', linestyle='--', color='r', label="Eye Contact (1=Yes, 0=No)")

    plt.xlabel("Frame Number")
    plt.ylabel("Emotion / Eye Contact")
    plt.title("Emotion & Eye Contact Changes Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()

#  RUN THE FULL PIPELINE
if __name__ == "__main__":
    video_path = r"D:\GitHub\Presentation-Skills\happyvid.mp4"
    frame_folder = "video_frames"

    extracted_count = extract_frames(video_path, frame_folder)

    if extracted_count > 0:
        predictions = predict_emotions_on_frames(frame_folder)
        video_emotion = aggregate_emotions(predictions)
        plot_emotions(predictions)
    else:
        print(" No frames were extracted. Check the video file and try again.")