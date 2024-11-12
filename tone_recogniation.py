import os
import librosa
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

tone_recog_path = r"C:\Users\MOUSTAFA\OneDrive\Desktop\tone_recog"
tone_detection_path = r"C:\\Users\\MOUSTAFA\\OneDrive\\Desktop\\tone_detection"

def load_data_from_dir(directory):
    data = []
    labels = []
    for label in os.listdir(directory):
        class_path = os.path.join(directory, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).mean(axis=1)
                    data.append(mfccs)
                    labels.append(label.lower())  
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(data), np.array(labels)


X_recog, y_recog = load_data_from_dir(tone_recog_path)
X_det, y_det = load_data_from_dir(tone_detection_path)


X = np.concatenate([X_recog, X_det], axis=0)
y = np.concatenate([y_recog, y_det], axis=0)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

def balance_data(X, y, encoder):
    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = min(counts)
    balanced_X = []
    balanced_y = []
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        selected_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_X.extend(X[selected_indices])
        balanced_y.extend(y[selected_indices])
    
    balanced_X = np.array(balanced_X)
    balanced_y = np.array(balanced_y)
    balanced_y_encoded = encoder.transform(balanced_y)
    return balanced_X, to_categorical(balanced_y_encoded)

X_balanced, y_balanced = balance_data(X, y, encoder)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
