import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
from collections import Counter


# Load PoseNet model from TensorFlow Hub
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(posenet_model_url)
print("PoseNet model loaded!")

# Access the correct signature for inference
infer = model.signatures['serving_default']  # This is the inference function


# Define body language meanings for postures
posture_meanings = {
    "Head Up": "Confidence",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Arms on Hips": "Confidence"
}

# Function to classify posture based on keypoints
def classify_posture(keypoints):
    # Example: Extract keypoints for important joints
    shoulder_y = keypoints[5]['y']  # Right shoulder
    hip_y = keypoints[11]['y']  # Right hip
    knee_y = keypoints[13]['y']  # Right knee
    ankle_y = keypoints[15]['y']  # Right ankle
    head_y = keypoints[0]['y']  # Nose (head)

    # **Head Up Posture**: Head is aligned with shoulders (neutral position)
    if abs(head_y - shoulder_y) < 0.05:  # Threshold can be adjusted
        return "Head Up"
    
    # **Slouching Posture**: Shoulders lower than hips and the spine is not straight.
    if shoulder_y > hip_y and knee_y > shoulder_y and ankle_y > knee_y:
        return "Slouching"
    
    # **Leaning Forward**: Head and shoulders leaning forward
    if head_y < shoulder_y and knee_y > hip_y:
        return "Leaning Forward"

    # **Head Down**: Head lower than shoulders
    if head_y > shoulder_y:
        return "Head Down"
    
    # **Leaning Back**: Shoulders higher than hips and knees
    if shoulder_y < hip_y and knee_y < hip_y and ankle_y < knee_y:
        return "Leaning Back"
    
    # **Arms on Hips Posture**: Shoulders and hips are in line but arms are placed on hips
    if abs(keypoints[5]['x'] - keypoints[6]['x']) < 0.1 and abs(keypoints[11]['x'] - keypoints[12]['x']) < 0.1:
        return "Arms on Hips"

    return "Unknown Posture"




# Function to extract keypoints from PoseNet output
def extract_keypoints(results):
    print("Shape of the output tensor:", results['output_0'].shape)
    keypoints = []
    for i in range(17):  # PoseNet detects 17 keypoints
        x = results['output_0'][0][0][i][1].numpy()
        y = results['output_0'][0][0][i][0].numpy()
        confidence = results['output_0'][0][0][i][2].numpy()
        print(f"Keypoint {i}: x={x}, y={y}, confidence={confidence}")
        keypoints.append({
            "x": x,
            "y": y,
            "confidence": confidence
        })
    return keypoints


# Initialize posture counter
posture_counter = {
    "Head Up": 0,
    "Slouching": 0,
    "Leaning Forward": 0,
    "Head Down": 0,
    "Leaning Back": 0,
    "Arms on Hips": 0,
    "Unknown Posture": 0
}


use_webcam = False  # Set to False if you want to use a video file
if use_webcam:
    cap = cv2.VideoCapture(0)  # Use the first webcam
    output_video_path = "webcam_output.mp4"
else:
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/TasnimGesture.mp4"  # Change this to your video file path
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/MayarGesture.mp4"  # Change this to your video file path
    input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/TedTalk.mp4"  # Change this to your video file path
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/TedTalk2.mp4"  # Change this to your video file path
    cap = cv2.VideoCapture(input_video_path)  # Open the video file

paused = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# Resize the frame to 192x192 as required by PoseNet
    frame_resized = cv2.resize(frame, (192, 192))
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert frame to RGB (required by PoseNet)
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Convert frame to RGB (required by PoseNet)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.int32)

    results = infer(rgb_frame)  # Get the output from PoseNet

# Debug: Print the results from PoseNet
    print("Results from PoseNet:", results)

    # Extract keypoints from PoseNet output
    keypoints = extract_keypoints(results)
    
    # Classify the posture based on keypoints
    posture = classify_posture(keypoints)
    
    # Increment posture counter
    if posture in posture_counter:
        posture_counter[posture] += 1
    else:
        posture_counter["Unknown Posture"] += 1

    
    # Get the posture and gesture meanings
    posture_meaning = posture_meanings.get(posture, "Unknown Meaning")
    # gesture_meaning = posture_meanings.get(gesture, "Unknown Meaning")

# Draw bounding box around keypoints
    for keypoint in keypoints:
        if keypoint['confidence'] > 0.5:  # Only draw if confidence is high enough
            x = int(keypoint['x'] * frame.shape[1])
            y = int(keypoint['y'] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Highlight posture area (head, shoulders, etc.)
    shoulder_x = int(keypoints[5]['x'] * frame.shape[1])
    shoulder_y = int(keypoints[5]['y'] * frame.shape[0])
    head_x = int(keypoints[0]['x'] * frame.shape[1])
    head_y = int(keypoints[0]['y'] * frame.shape[0])

   


    # Display the detected posture and its meaning
    cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Posture Meaning: {posture_meaning}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the video feed with posture and gesture analysis
    cv2.imshow("PoseNet Posture and Gesture Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if key == ord('q'):
        break

    # Press 'SPACE' to pause/resume
    if key == ord(' '):  
        paused = not paused  # Toggle pause state

     # **Wait until the user presses SPACE again to resume**
        while paused:
            key2 = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
            if key2 == ord(' '):  # Resume when SPACE is pressed again
                paused = False
                break
            elif key2 == ord('q'):  # Quit if 'q' is pressed while paused
                cap.release()
                cv2.destroyAllWindows()
                exit()   

cap.release()
cv2.destroyAllWindows()

# Print posture counts with meanings
print("\nPosture Count Summary:")
for posture, count in posture_counter.items():
    meaning = posture_meanings.get(posture, "Unknown Meaning")
    print(f"{posture}: {count} times - Meaning: {meaning}")

