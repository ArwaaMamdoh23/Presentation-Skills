# Import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#  Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

#  Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
        r'D:\GitHub\Presentation-Skills\DataSet\train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

#  Preprocess all test images
validation_generator = validation_data_gen.flow_from_directory(
        r'D:\GitHub\Presentation-Skills\DataSet\validation',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

#  Create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(8, activation='softmax'))

#  Ensure OpenCL is disabled for OpenCV compatibility
cv2.ocl.setUseOpenCL(False)

#  Compile model
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

#  Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_generator))

#  Print Final Training & Validation Accuracy
train_acc = emotion_model_info.history['accuracy'][-1]
val_acc = emotion_model_info.history['val_accuracy'][-1]
print("\n🔹 Training Accuracy: {:.2f}%".format(train_acc * 100))
print("🔹 Validation Accuracy: {:.2f}%".format(val_acc * 100))

#  Save model structure in JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

#  Save trained model weights
emotion_model.save_weights("emotion_model.weights.h5")  # Saves only weights
emotion_model.save('emotion_model.keras')  # Saves full model

print("\n Model and weights saved successfully!")
