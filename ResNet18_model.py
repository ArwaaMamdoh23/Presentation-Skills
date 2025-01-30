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


# Define a transformation for image preprocessing
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
model = GestureBodyLanguageModel(num_gesture_classes, num_body_language_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
gesture_loss_fn = nn.CrossEntropyLoss()
body_language_loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training loop
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    correct_gesture_preds_train = 0
    correct_body_language_preds_train = 0
    total_train = 0
    # running_gesture_loss = 0.0
    # running_body_language_loss = 0.0
    


    for gesture_images, body_images, gesture_labels, body_language_labels in train_loader:
        # Move data to GPU if available
        gesture_images, body_images = gesture_images.to(device), body_images.to(device)
        gesture_labels, body_language_labels = gesture_labels.to(device), body_language_labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        gesture_preds, body_language_preds = model(gesture_images, body_images)

        gesture_loss = gesture_loss_fn(gesture_preds, gesture_labels.long())
        body_language_loss = body_language_loss_fn(body_language_preds, body_language_labels.long())


        # # Compute losses
        # gesture_loss = gesture_loss_fn(gesture_preds, gesture_labels)
        # body_language_loss = body_language_loss_fn(body_language_preds, body_language_labels)

        # Backpropagate and optimize
        total_loss = gesture_loss + body_language_loss
        total_loss.backward()
        optimizer.step()

# Calculate training accuracy
        _, gesture_pred_labels = torch.max(gesture_preds, 1)
        _, body_language_pred_labels = torch.max(body_language_preds, 1)

 # Update correct predictions
        correct_gesture_preds_train += (gesture_pred_labels == gesture_labels).sum().item()
        correct_body_language_preds_train += (body_language_pred_labels == body_language_labels).sum().item()
        total_train += gesture_labels.size(0)

# Calculate training accuracy
    gesture_accuracy_train = correct_gesture_preds_train / total_train * 100
    body_language_accuracy_train = correct_body_language_preds_train / total_train * 100

# Print accuracy for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Gesture Accuracy: {gesture_accuracy_train:.2f}%, Body Language Accuracy: {body_language_accuracy_train:.2f}%')


        # Accumulate the loss values
        # running_gesture_loss += gesture_loss.item()
        # running_body_language_loss += body_language_loss.item()

    # print(f'Epoch [{epoch+1}/{num_epochs}], Gesture Loss: {running_gesture_loss / len(train_loader):.4f}, Body Language Loss: {running_body_language_loss / len(train_loader):.4f}')

# Validation loop
model.eval()
correct_gesture_preds = 0
correct_body_language_preds = 0
total = 0

with torch.no_grad():
    for gesture_images, body_images, gesture_labels, body_language_labels in val_loader:
        gesture_images, body_images = gesture_images.to(device), body_images.to(device)
        gesture_labels, body_language_labels = gesture_labels.to(device), body_language_labels.to(device)

        # Forward pass
        gesture_preds, body_language_preds = model(gesture_images, body_images)

        # Get predicted labels
        _, gesture_pred_labels = torch.max(gesture_preds, 1)
        _, body_language_pred_labels = torch.max(body_language_preds, 1)

        # Update correct predictions
        correct_gesture_preds += (gesture_pred_labels == gesture_labels).sum().item()
        correct_body_language_preds += (body_language_pred_labels == body_language_labels).sum().item()
        total += gesture_labels.size(0)

        # # Display images and predictions
        # for i in range(gesture_images.size(0)):
        #     # Convert tensor back to image format for display
        #     gesture_img = gesture_images[i].cpu().numpy().transpose(1, 2, 0)
        #     body_img = body_images[i].cpu().numpy().transpose(1, 2, 0)

        #     # Rescale images back to [0, 1] for proper display
        #     gesture_img = np.clip(gesture_img, 0, 1)
        #     body_img = np.clip(body_img, 0, 1)

        #     # Get the predicted labels as strings
        #     gesture_label = train_dataset.gesture_label_encoder.inverse_transform([gesture_pred_labels[i].item()])[0]
        #     body_language_label = train_dataset.body_language_label_encoder.inverse_transform([body_language_pred_labels[i].item()])[0]

            # # Plot the gesture image
            # plt.figure(figsize=(6, 3))
            # plt.subplot(1, 2, 1)
            # plt.imshow(gesture_img)
            # plt.axis('off')
            # plt.title(f"Predicted Gesture: {gesture_label}")

            # # Plot the body image
            # plt.subplot(1, 2, 2)
            # plt.imshow(body_img)
            # plt.axis('off')
            # plt.title(f"Predicted Body Language: {body_language_label}")

            # plt.show()




# Print validation accuracy
gesture_accuracy = correct_gesture_preds / total * 100
body_language_accuracy = correct_body_language_preds / total * 100
print(f'Validation Accuracy - Gesture: {gesture_accuracy:.2f}%, Body Language: {body_language_accuracy:.2f}%')




# # # Define the Dual-Branch Model

# # # Define a Dataset Class for Gesture and Body Language Data
# # class GestureBodyLanguageDataset(Dataset):
# #     def __init__(self, gesture_folder, body_folder, annotations, transform=None):
# #         self.gesture_folder = gesture_folder
# #         self.body_folder = body_folder
# #         self.annotations = annotations
# #         self.transform = transform
        
# #         # Initialize LabelEncoders to convert string labels into integers
# #         self.gesture_label_encoder = LabelEncoder()
# #         self.body_language_label_encoder = LabelEncoder()
        
# #         # Fit the encoders on the unique labels
# #         self.gesture_label_encoder.fit(self.annotations["class"])
# #         self.body_language_label_encoder.fit(self.annotations["body_language_label"])

# #     def __len__(self):
# #         return len(self.annotations)

# #     def __getitem__(self, idx):
# #         # Load gesture image
# #         gesture_path = os.path.join(self.gesture_folder, self.annotations.iloc[idx]["filename"])
# #         gesture_image = Image.open(gesture_path).convert("RGB")

# #         # Apply transformations
# #         if self.transform:
# #             gesture_image = self.transform(gesture_image)

# #         # Load labels and convert them to tensor using encoders
# #         gesture_label = self.annotations.iloc[idx]["class"]
# #         body_language_label = self.annotations.iloc[idx]["body_language_label"]

# #         # Convert the labels to integer indices
# #         gesture_label = self.gesture_label_encoder.transform([gesture_label])[0]  # Encode gesture label
# #         body_language_label = self.body_language_label_encoder.transform([body_language_label])[0]  # Encode body language label

# #         # Return the image and the encoded labels as tensors
# #         return gesture_image, tensor(gesture_label), tensor(body_language_label)


# # class GestureBodyLanguageModel(nn.Module):
# #     def __init__(self, num_gesture_classes, num_body_language_classes):
# #         super(GestureBodyLanguageModel, self).__init__()
# #         # Pretrained ResNet18 for gesture recognition
# #         self.gesture_branch = models.resnet18(pretrained=True)
# #         self.gesture_branch.fc = nn.Linear(self.gesture_branch.fc.in_features, 256)

# #         # Pretrained ResNet18 for body language context
# #         self.body_branch = models.resnet18(pretrained=True)
# #         self.body_branch.fc = nn.Linear(self.body_branch.fc.in_features, 256)

# #         # Fusion layers
# #         self.fc1 = nn.Linear(256 * 2, 512)
# #         self.fc2 = nn.Linear(512, 256)

# #         # Output layers
# #         self.gesture_out = nn.Linear(256, num_gesture_classes)
# #         self.body_language_out = nn.Linear(256, num_body_language_classes)

# #     def forward(self, x_gesture, x_body):
# #         # Extract features from each branch
# #         gesture_features = self.gesture_branch(x_gesture)
# #         body_features = self.body_branch(x_body)

# #         # Fuse features
# #         fused_features = torch.cat((gesture_features, body_features), dim=1)
# #         fused_features = F.relu(self.fc1(fused_features))
# #         fused_features = F.relu(self.fc2(fused_features))

# #         # Output predictions
# #         gesture_pred = self.gesture_out(fused_features)
# #         body_language_pred = self.body_language_out(fused_features)

# #         return gesture_pred, body_language_pred

# # # Define a Dataset Class for Gesture and Body Language Data
# # class GestureBodyLanguageDataset(Dataset):
# #     def __init__(self, gesture_folder, body_folder, annotations, transform=None):
# #         self.gesture_folder = gesture_folder
# #         self.body_folder = body_folder
# #         self.annotations = annotations
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.annotations)

# #     def __getitem__(self, idx):
# #         # Load gesture image
# #         gesture_path = os.path.join(self.gesture_folder, self.annotations.iloc[idx]["filename"])
# #         gesture_image = Image.open(gesture_path).convert("RGB")

# #         # Load body image
# #         body_path = os.path.join(self.body_folder, self.annotations.iloc[idx]["filename"])
# #         body_image = Image.open(body_path).convert("RGB")

# #         # Apply transformations
# #         if self.transform:
# #             gesture_image = self.transform(gesture_image)
# #             body_image = self.transform(body_image)

# #         # Load labels
# #         gesture_label = self.annotations.iloc[idx]["class"]
# #         body_language_label = self.annotations.iloc[idx]["body_language_label"]

# #         return gesture_image, body_image, gesture_label, body_language_label

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load data
# import pandas as pd

# annotations = pd.read_csv("D:/cnn_project/Dataset/train/_annotations.csv")  # Annotations file with gesture/body files and labels
# # Print the first few rows
# print(annotations.head())

# # Define the gesture-to-body language mapping
# gesture_to_body_language = {
#     "closedFist": "Aggressive, determined, or ready for action",
#     "fingerSymbols": "Thoughtful, symbolic, or communicative",
#     "semiOpenFist": "Neutral, relaxed, or slightly defensive",
#     "semiOpenPalm": "Friendly, open, or inviting",
#     "openPalm": "Honest, welcoming, or submissive",
#     "singleFingerBend": "Curious, questioning, or uncertain",
#     "fingerCircle": "Approval, agreement, or emphasis",
# }

# # Apply mapping to create a new 'body_language_label' column
# annotations["body_language_label"] = annotations["class"].map(gesture_to_body_language)

# # Display the first few rows with the new column
# print(annotations.head())

# # gesture_folder = "D:/cnn_project/Dataset/train"
# dataset = GestureBodyLanguageDataset("D:/cnn_project/Dataset/train", "D:/cnn_project/Dataset/train", annotations, transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Initialize the model
# num_gesture_classes = len(annotations["class"].unique())
# num_body_language_classes = len(annotations["body_language_label"].unique())

# model = GestureBodyLanguageModel(num_gesture_classes, num_body_language_classes)

# # Define optimizer and loss functions
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# gesture_loss_fn = nn.CrossEntropyLoss()
# body_language_loss_fn = nn.CrossEntropyLoss()

# # Training loop
# for epoch in range(10):
#     model.train()
#     total_gesture_loss = 0.0
#     total_body_language_loss = 0.0

#     for gesture_images, body_images, gesture_labels, body_language_labels in dataloader:
#         # Move data to device (GPU or CPU)
#         gesture_images = gesture_images
#         body_images = body_images
#         gesture_labels = gesture_labels
#         body_language_labels = body_language_labels

#         # Forward pass
#         gesture_preds, body_language_preds = model(gesture_images, body_images)

#         # Compute losses
#         gesture_loss = gesture_loss_fn(gesture_preds, gesture_labels)
#         body_language_loss = body_language_loss_fn(body_language_preds, body_language_labels)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss = gesture_loss + body_language_loss
#         loss.backward()
#         optimizer.step()

#         # Track losses
#         total_gesture_loss += gesture_loss.item()
#         total_body_language_loss += body_language_loss.item()

#     print(f"Epoch {epoch + 1}, Gesture Loss: {total_gesture_loss:.4f}, Body Language Loss: {total_body_language_loss:.4f}")

# print("\n✅ Training complete! Model is ready to classify gestures and body language.\n")
