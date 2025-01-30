import cv2
import torch
import numpy as np
from torchvision import transforms
import mediapipe as mp
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from torch import tensor
from sklearn.preprocessing import LabelEncoder
from torch import optim
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

# Define transformation for video frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# GestureBodyLanguageDataset (with label encoding)
class GestureBodyLanguageDataset(Dataset):
    def __init__(self, gesture_folder, body_folder, annotations, transform=None):
        self.gesture_folder = gesture_folder
        self.body_folder = body_folder
        self.annotations = annotations
        self.transform = transform

        # Initialize LabelEncoders to convert string labels into integers
        self.gesture_label_encoder = LabelEncoder()
        self.body_language_label_encoder = LabelEncoder()

        # Fit the encoders on the unique labels
        self.gesture_label_encoder.fit(self.annotations["class"])
        self.body_language_label_encoder.fit(self.annotations["body_language_label"])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load gesture image
        gesture_path = os.path.join(self.gesture_folder, self.annotations.iloc[idx]["filename"])
        gesture_image = Image.open(gesture_path).convert("RGB")

        # Load body image
        body_path = os.path.join(self.body_folder, self.annotations.iloc[idx]["filename"])
        body_image = Image.open(body_path).convert("RGB")

        # Apply transformations
        if self.transform:
            gesture_image = self.transform(gesture_image)
            body_image = self.transform(body_image)

        # Load labels
        gesture_label = self.annotations.iloc[idx]["class"]
        body_language_label = self.annotations.iloc[idx]["body_language_label"]

        # Convert labels to integer indices using encoders
        gesture_label = self.gesture_label_encoder.transform([gesture_label])[0]
        body_language_label = self.body_language_label_encoder.transform([body_language_label])[0]

        return gesture_image, body_image, gesture_label, body_language_label


# GestureBodyLanguageModel (with two branches)
class GestureBodyLanguageModel(nn.Module):
    def __init__(self, num_gesture_classes, num_body_language_classes):
        super(GestureBodyLanguageModel, self).__init__()

        # Pretrained ResNet18 for gesture recognition
        self.gesture_branch = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.gesture_branch.fc = nn.Linear(self.gesture_branch.fc.in_features, 256)

        # Pretrained ResNet18 for body language context
        self.body_branch = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.body_branch.fc = nn.Linear(self.body_branch.fc.in_features, 256)

        # Fusion layers to combine features
        self.fc1 = nn.Linear(256 * 2, 512)
        self.fc2 = nn.Linear(512, 256)

        # Output layers for predictions
        self.gesture_out = nn.Linear(256, num_gesture_classes)
        self.body_language_out = nn.Linear(256, num_body_language_classes)

    def forward(self, x_gesture, x_body):
        # Extract features from both branches
        gesture_features = self.gesture_branch(x_gesture)
        body_features = self.body_branch(x_body)

        # Concatenate the features from both branches
        fused_features = torch.cat((gesture_features, body_features), dim=1)
        fused_features = F.relu(self.fc1(fused_features))
        fused_features = F.relu(self.fc2(fused_features))

        # Generate predictions for gesture and body language
        gesture_pred = self.gesture_out(fused_features)
        body_language_pred = self.body_language_out(fused_features)

        return gesture_pred, body_language_pred


# Dataset and DataLoader setup
gesture_folder = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/GestureDataset/train'
body_folder = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/GestureDataset/train'
annotations_file = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/GestureDataset/train/_annotations.csv'

annotations = pd.read_csv(annotations_file)

gesture_to_body_language = {
    "closedFist": "Aggressive, determined, or ready for action",
    "fingerSymbols": "Thoughtful, symbolic, or communicative",
    "semiOpenFist": "Neutral, relaxed, or slightly defensive",
    "semiOpenPalm": "Friendly, open, or inviting",
    "openPalm": "Honest, welcoming, or submissive",
    "singleFingerBend": "Curious, questioning, or uncertain",
    "fingerCircle": "Approval, agreement, or emphasis",
}

# Apply mapping to create a new 'body_language_label' column
annotations["body_language_label"] = annotations["class"].map(gesture_to_body_language)

# Split dataset into train and validation sets
train_data, val_data = train_test_split(annotations, test_size=0.2)

# Create datasets and DataLoaders
train_dataset = GestureBodyLanguageDataset(gesture_folder, body_folder, train_data, transform=transform)
val_dataset = GestureBodyLanguageDataset(gesture_folder, body_folder, val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model and optimizer
num_gesture_classes = len(train_dataset.gesture_label_encoder.classes_)
num_body_language_classes = len(train_dataset.body_language_label_encoder.classes_)


# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureBodyLanguageModel(num_gesture_classes, num_body_language_classes).to(device)
model.load_state_dict(torch.load("D:\cnn_project\model.pth", map_location=device))  # Load trained model
model.eval()


# Initialize MediaPipe Hands & Pose detection
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Load video file (replace with your video path)
video_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/TasnimGesture.mp4"
cap = cv2.VideoCapture(video_path)

# Get video frame rate and dimensions
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer to save the processed video
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Convert frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # Detect hands
    hand_results = hands.process(rgb_frame)

    # Detect body pose
    pose_results = pose.process(rgb_frame)

    gesture_img, body_img = None, None

    # Extract hand region if detected
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img_w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img_h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img_w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img_h)

            gesture_img = frame[y_min:y_max, x_min:x_max]
            break  # Only process one hand

    # Extract full-body image if pose detected
    if pose_results.pose_landmarks:
        body_img = frame  # Use entire frame as body language context

    # Process images if both are available
    if gesture_img is not None and body_img is not None:
        # Convert OpenCV images to PIL format for transformation
        gesture_pil = Image.fromarray(cv2.cvtColor(gesture_img, cv2.COLOR_BGR2RGB))
        body_pil = Image.fromarray(cv2.cvtColor(body_img, cv2.COLOR_BGR2RGB))

        # Apply transformations
        gesture_tensor = transform(gesture_pil).unsqueeze(0).to(device)
        body_tensor = transform(body_pil).unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            gesture_pred, body_language_pred = model(gesture_tensor, body_tensor)

        # Get predicted class labels
        gesture_label = train_dataset.gesture_label_encoder.inverse_transform(
            [gesture_pred.argmax(dim=1).item()]
        )[0]

        body_language_label = train_dataset.body_language_label_encoder.inverse_transform(
            [body_language_pred.argmax(dim=1).item()]
        )[0]

        # Display Predictions on Video
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Body Language: {body_language_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write processed frame to output video
    out.write(frame)

    # Show frame (optional)
    cv2.imshow("Gesture & Body Language Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    correct_gesture = 0
    correct_body_language = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for gesture_img, body_img, gesture_label, body_language_label in dataloader:
            # Move data to device (GPU/CPU)
            gesture_img, body_img = gesture_img.to(device), body_img.to(device)
            gesture_label, body_language_label = gesture_label.to(device), body_language_label.to(device)

            # Forward pass
            gesture_pred, body_language_pred = model(gesture_img, body_img)
            
            # Get predicted class labels
            _, gesture_pred_label = torch.max(gesture_pred, 1)
            _, body_language_pred_label = torch.max(body_language_pred, 1)
            
            # Update correct counts
            correct_gesture += (gesture_pred_label == gesture_label).sum().item()
            correct_body_language += (body_language_pred_label == body_language_label).sum().item()
            total += gesture_label.size(0)

    gesture_accuracy = correct_gesture / total * 100
    body_language_accuracy = correct_body_language / total * 100
    return gesture_accuracy, body_language_accuracy

# Evaluate the model on validation data
gesture_accuracy, body_language_accuracy = calculate_accuracy(model, val_loader, device)
print(f"Gesture Accuracy: {gesture_accuracy:.2f}%")
print(f"Body Language Accuracy: {body_language_accuracy:.2f}%")



print(f"Processed video saved at: {output_path}")
