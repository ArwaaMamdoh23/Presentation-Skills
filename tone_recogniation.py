import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

dataset_path = r"D:\Senior\grad project\tone detection\tone detection"

categories = ['Men_angry', 'Men_Sad', 'Men_neutral', 'Men_fear', 'Men_happy', 
              'Female_angry', 'Female_fear', 'Female_happy', 'Female_sad', 'Female_neutral']

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def load_data(dataset_path):
    features = []
    labels = []
    label_encoder = LabelEncoder()
    
    for category in categories:
        file_path = os.path.join(dataset_path, category, '*.wav')
        files = glob.glob(file_path)
        
        for file in files:
            features.append(extract_features(file))
            labels.append(category)
    
    features = np.array(features)
    labels = label_encoder.fit_transform(labels)
    return features, labels, label_encoder

features, labels, label_encoder = load_data(dataset_path)

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

def predict_emotion(audio_path, model, scaler, label_encoder):
    feature = extract_features(audio_path)
    print("Extracted Features:", feature)
    feature_scaled = scaler.transform([feature])
    prediction = model.predict(feature_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)
    decision = model.decision_function(feature_scaled)
    print("Decision Function Output:", decision)
    return predicted_label[0]

audio_file_path = r"C:\Users\MOUSTAFA\OneDrive\Desktop\female_neutral (1).wav"

predicted_emotion = predict_emotion(audio_file_path, svm_model, scaler, label_encoder)

print(f"Predicted Emotion: {predicted_emotion}")

audio, sr = librosa.load(audio_file_path, sr=22050)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title("MFCC Features of the Test Audio")
plt.show()
