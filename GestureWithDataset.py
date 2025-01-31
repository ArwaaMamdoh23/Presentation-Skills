import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/GestureDataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")
labels_file = os.path.join(train_path, "annotations.csv")  # Update with your actual labels file
# Load labels (Modify if using JSON)
df = pd.read_csv(labels_file)  # Read CSV containing image-label pairs

# Ensure columns are as expected
print("Columns in CSV:", df.columns)

# Split the data (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to move images into class folders
def organize_images(data_path, df):
    for _, row in df.iterrows():
        image_name, label = row["filename"], row["class"]  # Update column names if different
        class_folder = os.path.join(data_path, label)

        # Create class folder if it doesn't exist
        os.makedirs(class_folder, exist_ok=True)

        # Move image to its respective class folder
        image_path = os.path.join(data_path, image_name)
        if os.path.exists(image_path):  # Check if image exists
            shutil.move(image_path, os.path.join(class_folder, image_name))
        else:
            print(f"⚠ Image not found: {image_name}")

# Organize train and test datasets
organize_images(train_path, df[df["split"] == "train"])  # Filter train images
organize_images(test_path, df[df["split"] == "test"])  # Filter test images

print("✅ Dataset organized successfully!")

# import torch
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images to 128x128
#     transforms.ToTensor(),  # Convert images to tensors
#     transforms.Normalize([0.5], [0.5])  # Normalize pixel values
# ])

# # Set dataset path (update this to where you extracted the dataset)
# dataset_path = "D:/4th Year 1st Term/Graduation Project/Presentation-Skills/GestureDataset/train"

# # Load dataset from folder (each subfolder is a class)
# gesture_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# # Create a DataLoader for training
# gesture_loader = DataLoader(gesture_dataset, batch_size=32, shuffle=True)

# # Print available gesture classes
# print("Loaded Gesture Classes:", gesture_dataset.classes)
