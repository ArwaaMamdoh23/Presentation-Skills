import os
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

# Enable mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

# Paths to the combined datasets
combined_train_path = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Trained_Dataset'
combined_validation_path = 'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Validation_Dataset'

# Optimized data augmentation
train_data_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

validation_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load VGG19 model without top layers
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze most VGG19 layers
for layer in vgg19_base.layers[:-6]:  # Unfreeze only the last 6 layers
    layer.trainable = False

# Custom classification layers
x = GlobalAveragePooling2D()(vgg19_base.output)
x = Dense(256, activation='relu')(x)  # Reduced from 512 to 256 for faster training
x = Dropout(0.6)(x)
predictions = Dense(8, activation='softmax', dtype='float32')(x)  # Ensure output in float32

# Final model
emotion_model = Model(inputs=vgg19_base.input, outputs=predictions)

# Compile model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Optimized batch size
batch_size = 32

# Data generators
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
    color_mode='rgb',
    class_mode='categorical'
)

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Final Training & Validation Accuracy 
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
