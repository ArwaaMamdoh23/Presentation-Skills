import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Optional: for visualization

# **Dictionary to Count Gesture Frequency**
gesture_counter = Counter()
# **Track last counted frame for each gesture**
last_counted_frame = {}

# Frame interval for counting (every 10 frames)
FRAME_INTERVAL = 60
current_frame = 0  # Track the current frame number

# Gesture to Body Language Mapping
gesture_to_body_language = {
    "Open Palm 🖐️": "Honesty",
    "Closed Fist ✊": "Determination",
    "Pointing Finger ☝️": "Confidence",
    "Thumbs Up 👍": "Encouragement",
    "Thumbs Down 👎": "Criticism",
    "Victory Sign ✌️": "Peace",
    "OK Sign 👌": "Agreement",
    "Rock Sign 🤘": "Excitement",
    "Call Me 🤙": "Friendly"
}

# **Create a Gesture Report file**
gesture_report_path = "Gesture_Report.txt"
with open(gesture_report_path, "w", encoding="utf-8") as file:
    file.write("📝 Gesture Report\n")
    file.write("=" * 30 + "\n")

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

    
    thumb_extended = landmarks[THUMB_TIP][1] < landmarks[THUMB_IP][1]
    # thumb_extended = landmarks[THUMB_TIP][0] > landmarks[THUMB_MCP][0]  # Thumb extended outward
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
    
# ✅ Thumbs Down Detection (Thumb down and other fingers curled)
    if not thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Thumbs Down 👎"

    if not thumb_extended and all([index_extended, middle_extended]) and not any([ring_extended, pinky_extended]):
        return "Victory Sign ✌️"

    if thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        return "OK Sign 👌"

    if index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        return "Rock Sign 🤘"

    if thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        return "Call Me 🤙"

    return "Unknown Gesture"

# **🔹 Read input video file instead of webcam**
use_webcam = False  # Set to False if you want to use a video file
if use_webcam:
    cap = cv2.VideoCapture(0)  # Use the first webcam
    output_video_path = "webcam_output.mp4"
else:
    input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"  # Change this to your video file path
    # input_video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/MayarGesture.mp4"  # Change this to your video file path
    cap = cv2.VideoCapture(input_video_path)
    output_video_path = "output_video.mp4"

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Save the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

paused = False  # Variable to track pause state

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    current_frame += 1  # Increment frame counter

# Rotate frame if needed
    # if frame_width > frame_height:
    #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # elif frame_height > frame_width and frame[0, 0, 0] < frame[-1, -1, 0]:
    #     frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert to RGB (MediaPipe requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = mp_hands.process(rgb_frame)

    # Check for detected hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Classify gesture
            gesture_name = classify_hand_gesture(hand_landmarks)
# ✅ Skip unknown gestures
            if gesture_name == "Unknown Gesture":
                continue  # 🚀 Skips counting, logging, and displaying Unknown Gesture
            
# ✅ Count the gesture **only if 10 frames have passed since the last count**
            if gesture_name not in last_counted_frame or (current_frame - last_counted_frame[gesture_name]) >= FRAME_INTERVAL:
                gesture_counter[gesture_name] += 1  # Count gesture
                last_counted_frame[gesture_name] = current_frame  # Update last counted frame

            # Get body language meaning
            # body_language_meaning = gesture_to_body_language.get(gesture_name, "Unknown Meaning")
            # print(f"🔍 Detected Gesture: {gesture_name}")
            body_language_meaning = gesture_to_body_language.get(gesture_name, "Unknown Meaning")
            # print(f"📝 Meaning: {body_language_meaning}")

            # Draw hand landmarks (optional)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

# ✅ Count detected gestures
            gesture_counter[gesture_name] += 1

            # Get bounding box coordinates
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            gesture_display = gesture_name if gesture_name in gesture_to_body_language else "Unknown Gesture"
            body_language_display = body_language_meaning if body_language_meaning != "Unknown Meaning" else "No Interpretation"
            # Display gesture label
            cv2.putText(frame, f"Gesture: {gesture_name}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Display body language meaning
            cv2.putText(frame, f"Body: {body_language_meaning}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# ✅ **Save gesture detection to file with timestamp**
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(gesture_report_path, "a", encoding="utf-8") as file:
                file.write(f"[{timestamp}] Detected Gesture: {gesture_name} - Meaning: {body_language_meaning}\n")

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


# ✅ Print & Save the Most Repeated Gestures
most_common_gestures = gesture_counter.most_common(5)
print("\n📊 Most Repeated Gestures:")
with open(gesture_report_path, "a", encoding="utf-8") as file:
    file.write("\n📊 Most Repeated Gestures:\n")

for gesture, count in most_common_gestures:
    meaning = gesture_to_body_language.get(gesture, "Unknown Meaning")
    print(f"{gesture} ({meaning}): {count} times")
    with open(gesture_report_path, "a", encoding="utf-8") as file:
        file.write(f"{gesture} ({meaning}): {count} times\n")

cap.release()
out.release()
cv2.destroyAllWindows()


print(f"Processed video saved at: {output_video_path}")
print(f"Gesture report saved at: {gesture_report_path}")

