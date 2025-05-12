import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time
from collections import Counter


posenet_model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
model = hub.load(posenet_model_url)
print("PoseNet model loaded!")

infer = model.signatures['serving_default'] 
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

def classify_posture(keypoints):
    shoulder_y = keypoints[5]['y']  
    hip_y = keypoints[11]['y']  
    knee_y = keypoints[13]['y']  
    ankle_y = keypoints[15]['y']  
    head_y = keypoints[0]['y']  

    if abs(head_y - shoulder_y) < 0.05:  
        return "Head Up"
    
    
    if shoulder_y > hip_y and knee_y > shoulder_y and ankle_y > knee_y:
        return "Slouching"
    
    if head_y < shoulder_y and knee_y > hip_y:
        return "Leaning Forward"

    if head_y > shoulder_y:
        return "Head Down"
    
    if shoulder_y < hip_y and knee_y < hip_y and ankle_y < knee_y:
        return "Leaning Back"
    
    if abs(keypoints[5]['x'] - keypoints[6]['x']) < 0.1 and abs(keypoints[11]['x'] - keypoints[12]['x']) < 0.1:
        return "Arms on Hips"
    if keypoints[9]['y'] < keypoints[3]['y'] and keypoints[10]['y'] < keypoints[4]['y']:
        return "Crossed Arms"


    if keypoints[9]['y'] > hip_y and keypoints[10]['y'] > hip_y:
        return "Hands in Pockets"

    return "Unknown Posture"




def extract_keypoints(results):
    print("Shape of the output tensor:", results['output_0'].shape)
    keypoints = []
    for i in range(17):  
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


posture_counter = {
    "Head Up": 0,
    "Slouching": 0,
    "Leaning Forward": 0,
    "Head Down": 0,
    "Leaning Back": 0,
    "Arms on Hips": 0,
    "Hands in Pockets": 0,
    "Crossed Arms": 0
}


use_webcam = False 
if use_webcam:
    cap = cv2.VideoCapture(0)  
    output_video_path = "webcam_output.mp4"
else:
    
    input_video_path = "Videos/Yehia.mp4"  
    cap = cv2.VideoCapture(input_video_path) 

paused = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    frame_resized = cv2.resize(frame, (192, 192))

    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    rgb_frame = tf.convert_to_tensor(rgb_frame, dtype=tf.int32)

    results = infer(rgb_frame)  

    print("Results from PoseNet:", results)

    keypoints = extract_keypoints(results)
    
    posture = classify_posture(keypoints)
    
    if posture != "Unknown Posture":
        posture_counter[posture] += 1
    
    posture_meaning = posture_meanings.get(posture, "Unknown Meaning")

    for keypoint in keypoints:
        if keypoint['confidence'] > 0.5:  
            x = int(keypoint['x'] * frame.shape[1])
            y = int(keypoint['y'] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    shoulder_x = int(keypoints[5]['x'] * frame.shape[1])
    shoulder_y = int(keypoints[5]['y'] * frame.shape[0])
    head_x = int(keypoints[0]['x'] * frame.shape[1])
    head_y = int(keypoints[0]['y'] * frame.shape[0])

   


    cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Posture Meaning: {posture_meaning}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("PoseNet Posture and Gesture Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to exit
    if key == ord('q'):
        break

    if key == ord(' '):  
        paused = not paused  

        while paused:
            key2 = cv2.waitKey(0) & 0xFF  
            if key2 == ord(' '):  
                paused = False
                break
            elif key2 == ord('q'):  
                cap.release()
                cv2.destroyAllWindows()
                exit()   

cap.release()
cv2.destroyAllWindows()

print("\nPosture Count Summary:")
for posture, count in posture_counter.items():
    meaning = posture_meanings.get(posture, "Unknown Meaning")
    print(f"{posture}: {count} times - Meaning: {meaning}")

