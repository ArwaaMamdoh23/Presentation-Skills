import cv2
import mediapipe as mp

# Initialize MediaPipe Hands & Pose
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.3)

# Gesture to Body Language Mapping (Basic Example)
gesture_to_body_language = {
    "Closed Fist": "Aggressive, determined, or ready for action",
    "Open Palm": "Honest, welcoming, or submissive",
    "Pointing Finger": "Directing, questioning, or emphasizing",
    "Thumbs Up": "Approval, positive, or encouragement",
    "Victory Sign": "Success, peace, or confidence",
}

# Load video
video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"
cap = cv2.VideoCapture(video_path)

# Get video frame properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when video ends

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert frame to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # Detect hands
    hand_results = mp_hands.process(rgb_frame)

    # Detect body pose
    pose_results = mp_pose.process(rgb_frame)

    # Initialize label variables
    gesture_label = "Unknown"
    body_language_label = "Unknown"

    # Draw Hand Landmarks & Bounding Box
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img_w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img_h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img_w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img_h)

            # Ensure valid bounding box
            if x_min < 0 or y_min < 0 or x_max > img_w or y_max > img_h:
                continue

            # Draw bounding box around hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Assign a dummy gesture label based on bounding box width (for demonstration)
            hand_width = x_max - x_min
            if hand_width > 150:
                gesture_label = "Open Palm"
            elif hand_width > 100:
                gesture_label = "Closed Fist"
            else:
                gesture_label = "Pointing Finger"

            # Get corresponding body language meaning
            body_language_label = gesture_to_body_language.get(gesture_label, "Neutral")

            # Display the gesture label above the bounding box
            cv2.putText(frame, f"Gesture: {gesture_label}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the body language meaning below the bounding box
            cv2.putText(frame, f"Body: {body_language_label}", (x_min, y_max + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Draw Pose Landmarks
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
            # cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    # Save processed frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow("Hand & Body Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key press

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")
