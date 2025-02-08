import cv2
import numpy as np
from keras.models import load_model
from keras import Model
from keras.layers import Conv2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- Step 1: Load the Trained Model ----------------
print("Loading trained model...")
emotion_model = load_model("emotion_model.keras")  # Load full model
print("Model loaded successfully!")

# Adjust the model to accept grayscale images
input_layer = Input(shape=(48, 48, 1))
x = Conv2D(3, (3, 3), padding='same')(input_layer)  # Convert 1 channel to 3 internally
output_layer = emotion_model(x)

# Create new model
new_model = Model(inputs=input_layer, outputs=output_layer)
new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------- Step 2: Evaluate Model on Test Data ----------------
# Initialize image data generator for test data
test_data_gen = ImageDataGenerator(rescale=1./255)

# Load test dataset
test_generator = test_data_gen.flow_from_directory(
    'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/DataSet/test',  # Path to test dataset
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",  # Using grayscale as the original dataset likely uses it
    class_mode='categorical',
    shuffle=False  # Important: Keep predictions aligned with labels
)

# Evaluate model performance
test_loss, test_acc = new_model.evaluate(test_generator, steps=len(test_generator))
print(f" Test Accuracy: {test_acc * 100:.2f}%")

# ---------------- Step 3: Real-Time Emotion Detection on Video ----------------
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad", 6: "Surprised", 7: "Other"}

# Load video for emotion detection
video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Videos/happyvid.mp4"
cap = cv2.VideoCapture(video_path)

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    # Read each frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and convert to grayscale
    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        # Extract the face region
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0) / 255.0  # Normalize

        # Predict the emotion
        emotion_prediction = new_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Display emotion label
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the output frame
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()