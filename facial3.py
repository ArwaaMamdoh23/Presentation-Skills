import os
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, UpSampling2D
from keras.applications import VGG19
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (48, 48, 3)
inputs = Input(shape=input_shape)

# First convolutional layer
x = Conv2D(128, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Second convolutional layer
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Apply MaxPooling with a smaller pool size to prevent shrinking too much
x = MaxPooling2D(pool_size=(2, 2))(x)

# Resize the feature map to (48, 48, 3) using UpSampling2D
x = UpSampling2D(size=(2, 2))(x)  # Upsample to (48, 48)

# Ensure that the number of channels is 3 (for RGB) before feeding it into VGG19
x = Conv2D(3, (1, 1), padding='same')(x)

# Load VGG19 base model
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Freeze VGG19 layers
for layer in vgg19_base.layers:
    layer.trainable = False

# Pass through VGG19
x = vgg19_base(x)

# Global average pooling
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)

# Final model
emotion_model = Model(inputs=inputs, outputs=predictions)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

batch_size = 64

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'D:/GitHub/Presentation-Skills/Trained_Dataset',
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'D:/GitHub/Presentation-Skills/Validation_Dataset',
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

for layer in vgg19_base.layers[-4:]:
    layer.trainable = True

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5), metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights("emotion_model.weights.h5")
emotion_model.save('emotion_model.keras')

print("\n Model and weights saved successfully!")
