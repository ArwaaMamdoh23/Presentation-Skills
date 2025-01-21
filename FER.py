import os
from PIL import Image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau



# Define the path to your dataset
dataset_path = 'Dataset/test'  # Adjust to your dataset path within the repo

# List of emotions (folder names)
emotions = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']

# Define the size for resizing images
image_size = (48, 48)

def load_images(input_dir):
    data = []  # List to hold the images
    labels = []  # List to hold the corresponding label
    # Loop over each emotion folder
    for emotion in emotions:
        emotion_folder = os.path.join(input_dir, emotion)  # Folder for each emotion
        label = emotions.index(emotion)  # Convert emotion to a numeric label
        
        # Get all image filenames in the folder
        all_images = os.listdir(emotion_folder)
        
        # Load all images without limiting the count
        for img_name in all_images:
            img_path = os.path.join(emotion_folder, img_name)  # Path to each image
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)  # Convert image to numpy array
                data.append(img_array)  # Add the image to the list
                labels.append(label)  # Add the label to the list
                print(f"Loaded {img_name} in {emotion} folder")
            except Exception as e:
                print(f"Error processing image {img_name}: {e}")
    
    return np.array(data), np.array(labels)

def resize_images(data, size=(48, 48)):
    resized_data = []
    errors = []  # List to track any errors
    for img_array in data:
        try:
            img = Image.fromarray(img_array)  # Convert numpy array back to image
            img = img.resize(size)  # Resize to specified dimensions
            resized_data.append(np.array(img))  # Convert resized image back to numpy array
        except Exception as e:
            errors.append(f"Error resizing image: {e}")
    
    if errors:
        print(f"Resizing completed with {len(errors)} errors.")
        for error in errors:
            print(error)
    else:
        print("All images have been resized successfully.")
    
    return np.array(resized_data)

# Load all images into lists
X, y = load_images(dataset_path)

# Resize the loaded images
X_resized = resize_images(X, image_size)

# Display the shape of the resized dataset
print(f"Resized dataset shape: {X_resized.shape}")

# Optionally, you can return or save the processed data and labels
def normalize_images(data):
    return data / 255.0  # Normalize to [0, 1]

# Normalize the resized images
X_normalized = normalize_images(X_resized)

# Reshape if needed for CNNs
X_final = X_normalized.reshape(X_normalized.shape[0], 48, 48, 1)  # Add channel dimension (1 for grayscale)

# mean = np.mean(X_final, axis=(0, 1, 2), keepdims=True)
# std = np.std(X_final, axis=(0, 1, 2), keepdims=True)
# X_final = (X_final - mean) / std

print(f"Loaded and normalized dataset with {X_final.shape[0]} samples")
print(f"Min pixel value: {X_normalized.min()}")
print(f"Max pixel value: {X_normalized.max()}")


##################################################

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=len(emotions))
y_test = to_categorical(y_test, num_classes=len(emotions))

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
# Create a data generator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the generator on the training data **after** splitting
datagen.fit(X_train)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


def build_model(input_shape, num_classes):
    model = models.Sequential()

    # First Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # #4  
    # model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPooling2D((2, 2)))
  
    #  # Global Average Pooling
    # model.add(layers.GlobalAveragePooling2D())

    # # Flatten the output from the convolutional layers
    model.add(layers.Flatten())

    # Fully connected (dense) layer
    model.add(layers.Dense(128, activation='relu'))
  

    # Output layer with softmax activation for multi-class classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(emotions)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Build the model
model = build_model(input_shape=(48, 48, 1), num_classes=num_classes)

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot the training accuracy over epochs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot the training loss over epochs
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()