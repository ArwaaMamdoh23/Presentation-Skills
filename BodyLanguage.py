import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Optional: for visualization

# Gesture to Body Language Mapping
gesture_to_body_language = {
    "Open Palm 🖐️": "Trust, honesty, openness",
    "Closed Fist ✊": "Power, anger, determination",
    "Pointing Finger ☝️": "Authority, instruction, warning",
    "Thumbs Up 👍": "Approval, success, encouragement",
    "Thumbs Down 👎": "Disapproval, rejection, criticism",
    "Victory Sign ✌️": "Peace, victory, confidence",
    "OK Sign 👌": "Agreement, everything is fine",
    "Rock Sign 🤘": "Excitement, rebellion, energy",
    "Call Me 🤙": "Connection, friendliness, relaxation",
    "Finger Heart 💕": "Love, affection, positivity"
}

# Gesture classification function
def classify_hand_gesture(hand_landmarks):
    """
    Detects common hand gestures based on MediaPipe landmarks.
    Returns the detected gesture name.
    """

    # Extract landmark coordinates
    landmarks = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

    # Finger landmarks indices
    THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
    THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP = 3, 7, 11, 15, 19
    THUMB_MCP = 2

    # Determine if fingers are extended or curled
    def is_finger_extended(tip, dip):
        return landmarks[tip][1] < landmarks[dip][1]  # y-coordinate: lower is higher

    thumb_extended = landmarks[THUMB_TIP][0] > landmarks[THUMB_MCP][0]  # Thumb extended outward
    index_extended = is_finger_extended(INDEX_TIP, INDEX_DIP)
    middle_extended = is_finger_extended(MIDDLE_TIP, MIDDLE_DIP)
    ring_extended = is_finger_extended(RING_TIP, RING_DIP)
    pinky_extended = is_finger_extended(PINKY_TIP, PINKY_DIP)

    # Gesture classification logic
    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        return "Open Palm 🖐️"

    if not any([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended]):
        return "Closed Fist ✊"

    if index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "Pointing Finger ☝️"

    if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Up 👍"

    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign ✌️"

    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign 👌"

    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign 🤘"

    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me 🤙"

    if thumb_extended and index_extended and np.linalg.norm(landmarks[4] - landmarks[8]) < 0.05:
        return "Finger Heart 💕"

    return "Unknown Gesture"

# **🔹 Read input video file instead of webcam**
# input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/MayarGesture.mp4"  # Change this to your video file path
input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save the output video
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

paused = False  # Variable to track pause state

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# Rotate frame if needed
    if frame_width > frame_height:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif frame_height > frame_width and frame[0, 0, 0] < frame[-1, -1, 0]:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = mp_hands.process(rgb_frame)

    # Check for detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Classify gesture
            gesture_name = classify_hand_gesture(hand_landmarks)

            # Get body language meaning
            body_language_meaning = gesture_to_body_language.get(gesture_name, "Unknown Meaning")

            # Draw hand landmarks (optional)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display gesture label
            cv2.putText(frame, f"Gesture: {gesture_name}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display body language meaning
            cv2.putText(frame, f"Body: {body_language_meaning}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Save processed frame to output video
    out.write(frame)

    # Show video preview
    cv2.imshow("Hand Gesture & Body Language Recognition", frame)

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
                out.release()
                cv2.destroyAllWindows()
                print(f"Processed video saved at: {output_video_path}")
                exit()   

cap.release()
out.release()
cv2.destroyAllWindows()


print(f"Processed video saved at: {output_video_path}")

