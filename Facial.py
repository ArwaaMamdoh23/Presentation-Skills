import os
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import EarlyStopping

# # Paths to the combined datasets 
# combined_train_path = r'D:\GitHub\Presentation-Skills\Trained_Dataset'  
# combined_validation_path = r'D:\GitHub\Presentation-Skills\Validation_Dataset'  
combined_train_path = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Trained_Dataset'
combined_validation_path = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Validation_Dataset'

# Initialize image data generator with optimized augmentation 
train_data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use VGG19's preprocessing
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

validation_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load VGG19 model without top layers
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze all VGG19 layers initially
for layer in vgg19_base.layers:
    layer.trainable = False

# Add custom classification layers
x = vgg19_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)  # Assuming 8 emotion categories
emotion_model = Model(inputs=vgg19_base.input, outputs=predictions)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
batch_size = 64


train_generator = train_data_gen.flow_from_directory(
    combined_train_path,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',  
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    combined_validation_path,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',  # VGG19 requires RGB input
    class_mode='categorical'
)

# Train the model with frozen VGG19 layers
emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

#  Fine-tune the top VGG19 layers
# Unfreeze the top 4 convolutional layers
for layer in vgg19_base.layers[-4:]:
    layer.trainable = True
    
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

# Fine-tuning the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,  # Additional epochs for fine-tuning
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Print Final Training & Validation Accuracy 
train_acc = emotion_model_info.history['accuracy'][-1]
val_acc = emotion_model_info.history['val_accuracy'][-1]
print("\n🔹 Training Accuracy: {:.2f}%".format(train_acc * 100))
print("🔹 Validation Accuracy: {:.2f}%".format(val_acc * 100))

# Save model structure and weights 
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights
emotion_model.save_weights("emotion_model.weights.h5")  # Saves only weights
emotion_model.save('emotion_model.keras')               # Saves full model

print("\nModel and weights saved successfully!")