import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Load PoseNet model from TensorFlow Hub
posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(posenet_model_url)
print("PoseNet model loaded!")

# Access the correct signature for inference
infer = model.signatures['serving_default']  # This is the inference function


# Define body language meanings for postures
posture_meanings = {
    "Standing": "Confidence",
    "Sitting": "Comfort",
    "Slouching": "Lack of confidence",
    "Leaning Forward": "Interest",
    "Head Down": "Lack of confidence",
    "Leaning Back": "Relaxation",
    "Pointing": "Authority, instruction, or direction.",
    "Thumbs Up": "Approval, success, encouragement.",
    "Thumbs Down": "Disapproval, rejection, criticism.",
    "Victory Sign": "Peace, victory, confidence.",
    "OK Sign": "Agreement, everything is fine.",
    "Rock Sign": "Excitement, rebellion, energy.",
    "Call Me": "Connection, friendliness, relaxation.",
    "Arms on Hips": "Confidence",
}

# Function to classify posture based on keypoints
def classify_posture(keypoints):
    # Example: Extract keypoints for important joints
    shoulder_y = keypoints[5]['y']  # Right shoulder
    hip_y = keypoints[11]['y']  # Right hip
    knee_y = keypoints[13]['y']  # Right knee
    ankle_y = keypoints[15]['y']  # Right ankle
    head_y = keypoints[0]['y']  # Nose (head)

    # **Standing Posture**
    if shoulder_y < hip_y:
        return "Standing"

    # **Sitting Posture**
    elif shoulder_y > hip_y:
        return "Sitting"

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
    if abs(keypoints[7]['x'] - keypoints[8]['x']) < 0.1 and abs(keypoints[6]['x'] - keypoints[9]['x']) < 0.1:
        return "Arms on Hips"

    return "Unknown Posture"


# Function to detect gestures based on keypoints
def classify_gesture(keypoints):
    # Ensure that only the available keypoints are used
    thumb_extended = keypoints[4]['y'] < keypoints[3]['y']  # Thumb extended upwards
    index_extended = keypoints[8]['y'] < keypoints[7]['y']  # Index finger extended
    middle_extended = keypoints[12]['y'] < keypoints[11]['y']  # Middle finger extended
    ring_extended = keypoints[16]['y'] < keypoints[15]['y']  # Ring finger extended
    
    # Only check pinky if it's part of the 17 keypoints (pinky is 20 and 19, but PoseNet only detects 17 keypoints)
    pinky_extended = keypoints[20]['y'] < keypoints[19]['y'] if len(keypoints) > 19 else False

    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up"
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down"
    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing"
    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Victory Sign"
    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign"
    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign"
    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me"
    
    return "Unknown Gesture"






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


use_webcam = False  # Set to False if you want to use a video file
if use_webcam:
    cap = cv2.VideoCapture(0)  # Use the first webcam
    output_video_path = "webcam_output.mp4"
else:
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"  # Change this to your video file path
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/MayarGesture.mp4"  # Change this to your video file path
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TedTalk.mp4"  # Change this to your video file path
    input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TedTalk2.mp4"  # Change this to your video file path
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
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
# Convert the frame to tf.int32 as required by PoseNet
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.int32)

    results = infer(rgb_frame)  # Get the output from PoseNet

# Debug: Print the results from PoseNet
    print("Results from PoseNet:", results)

    # Extract keypoints from PoseNet output
    keypoints = extract_keypoints(results)
    
    # Classify the posture based on keypoints
    posture = classify_posture(keypoints)
    
    # Classify the gesture based on keypoints
    # gesture = classify_gesture(keypoints)
    
    # Get the posture and gesture meanings
    posture_meaning = posture_meanings.get(posture, "Unknown Meaning")
    # gesture_meaning = posture_meanings.get(gesture, "Unknown Meaning")

    # Display the detected posture and its meaning
    cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Posture Meaning: {posture_meaning}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame, f"Gesture Meaning: {gesture_meaning}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

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
    # # Exit when 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()


