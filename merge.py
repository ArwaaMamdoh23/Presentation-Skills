import os
import shutil
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Paths to the validation datasets (define these correctly)
validation_dataset_1_path = r'D:\GitHub\Presentation-Skills\DataSet\validation'  # Original validation dataset
validation_dataset_2_path = r'D:\GitHub\Presentation-Skills\DataSet2\validation'  # Additional validation dataset

# Path where the combined validation dataset will be saved
combined_validation_path = r'D:\GitHub\Presentation-Skills\Validation_Dataset'

# Ensure combined validation dataset directory exists, if not, create it
if not os.path.exists(combined_validation_path):
    os.makedirs(combined_validation_path)

# Function to copy images to the combined directory
def merge_datasets(source_dir, target_dir):
    """
    This function merges images from the source dataset to the target directory.
    It resizes, converts to grayscale, and copies them to the combined directory.
    """
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it's a class directory
            target_class_path = os.path.join(target_dir, class_name)
            if not os.path.exists(target_class_path):
                os.makedirs(target_class_path)
            
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                
                if os.path.isfile(image_path):
                    # Read the image
                    img = cv2.imread(image_path)
                    
                    # Resize to 48x48 and convert to grayscale
                    img_resized = cv2.resize(img, (48, 48))
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    
                    # Save the image in the target directory
                    target_image_path = os.path.join(target_class_path, filename)
                    cv2.imwrite(target_image_path, img_gray)

# Merge validation datasets only
merge_datasets(validation_dataset_1_path, combined_validation_path)
merge_datasets(validation_dataset_2_path, combined_validation_path)

print("Validation datasets merged successfully!")