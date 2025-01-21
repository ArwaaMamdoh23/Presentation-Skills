import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from hsemotion.facial_emotions import HSEmotionRecognizer


# Load the pre-trained HSEmotion model
model_name = 'enet_b0_8_best_afew'  # Other options: 'enet_b2_8_best_vgaf'
fer = HSEmotionRecognizer(model_name=model_name, device='cpu')

  # Use 'cuda' for GPU if available


def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extract frames from a video at a specified frame rate.
    """
    if not os.path.exists(video_path):
        print(f"⚠ Error: Video file not found at {video_path}")
        return 0

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"FPS Detected: {fps}")

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


def predict_emotions_on_frames(frame_folder):
    """
    Runs the pre-trained model on each extracted frame.
    """
    results = {}

    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".jpg")])

    if not frame_files:
        print(" Error: No frames found in folder. Make sure the extraction step worked correctly.")
        return results

    for frame_file in frame_files:
        image_path = os.path.join(frame_folder, frame_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f" Warning: Unable to read {frame_file}. Skipping...")
            continue

        # Predict Emotion
        emotion, scores = fer.predict_emotions(image, logits=True)
        results[frame_file] = emotion
        print(f"{frame_file}: {emotion}")

    return results


def aggregate_emotions(predictions):
    """
    Finds the most common emotion across all frames.
    """
    if not predictions:
        print(" Error: No emotions detected. Make sure the video is valid and frames are extracted.")
        return "No Emotion Detected"

    emotion_counts = Counter(predictions.values())
    dominant_emotion = emotion_counts.most_common(1)[0][0]

    print(f" Most Common Emotion in Video: {dominant_emotion}")
    return dominant_emotion

#  OPTIONAL: PLOT EMOTIONS OVER TIME
def plot_emotions(predictions):
    """
    Plots the detected emotions over time.
    """
    if not predictions:
        print(" No emotions detected, skipping the graph.")
        return

    frame_numbers = list(range(len(predictions)))
    emotions = list(predictions.values())

    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, emotions, marker='o', linestyle='-', color='b', label="Emotion")
    plt.xlabel("Frame Number")
    plt.ylabel("Predicted Emotion")
    plt.title("Emotion Changes Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Set paths
    video_path = r"D:\GitHub\Presentation-Skills\fervideo.mp4"  # Your video path is set here
    frame_folder = "video_frames"

    # Extract frames from video
    extracted_count = extract_frames(video_path, frame_folder)

    if extracted_count > 0:
        # Run emotion detection on frames
        predictions = predict_emotions_on_frames(frame_folder)

        # Find the most common emotion in the video
        video_emotion = aggregate_emotions(predictions)

        # Optional: Visualize emotions over time
        plot_emotions(predictions)
    else:
        print(" No frames were extracted. Check the video file and try again.")
