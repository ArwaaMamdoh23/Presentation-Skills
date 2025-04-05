import torch
import os
from sklearn.model_selection import train_test_split

data_folder = "processed_data1"
target_length = 16000  
waveforms = []
labels = []

for filename in os.listdir(data_folder):
    if filename.endswith(".pt"):
        filepath = os.path.join(data_folder, filename)
        waveform, label = torch.load(filepath)

        waveform = waveform.squeeze(0)  

        if waveform.shape[0] < target_length:
            pad_length = target_length - waveform.shape[0]
            waveform = torch.cat([waveform, torch.zeros(pad_length)])

        elif waveform.shape[0] > target_length:
            waveform = waveform[:target_length]

        waveforms.append(waveform)
        labels.append(label)

X = torch.stack(waveforms)
y = torch.tensor(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

torch.save((X_train, y_train), "train_data.pt")
torch.save((X_val, y_val), "val_data.pt")

print("Dataset processed, split, and saved successfully!")

