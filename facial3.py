import os
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.applications import VGG19
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load VGG19 model without top layers
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze VGG19 layers to retain pre-trained features
for layer in vgg19_base.layers:
    layer.trainable = False

# Add additional CNN layers on top of VGG19
x = vgg19_base.output

# Extra Convolutional Layer 1
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Extra Convolutional Layer 2
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Global Average Pooling and Classification Layers
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)

# Final model
emotion_model = Model(inputs=vgg19_base.input, outputs=predictions)

# Compile the model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Batch size
batch_size = 64

# Data generators
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Trained_Dataset',
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'D:/4th Year 1st Term/Graduation Project/Presentation-Skills/Validation_Dataset',
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',
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

# Fine-tune the top VGG19 layers
for layer in vgg19_base.layers[-4:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

# Fine-tune the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

# Save the model
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights("emotion_model.weights.h5")
emotion_model.save('emotion_model.keras')

print("\n Model and weights saved successfully!")
