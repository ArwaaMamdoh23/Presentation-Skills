import torch
import torchaudio
import pandas as pd
import os
from torchaudio.transforms import Resample
from sklearn.model_selection import train_test_split

class PodcastFillersDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, audio_folder):
        self.annotations = pd.read_csv(csv_file)

        # Drop rows where label is NaN
        self.annotations = self.annotations.dropna(subset=['label_consolidated_vocab'])

        self.audio_folder = audio_folder

        # Generate label-to-ID mapping **after** filtering NaNs
        self.labels = self.annotations['label_consolidated_vocab'].unique()
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}

        print("Label to ID mapping:", self.label2id)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        clip_name = self.annotations.iloc[idx]['clip_name']
        audio_path = os.path.join(self.audio_folder, clip_name)
        
        # Load waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # 🔴 Ensure correct sampling rate for Wav2Vec2 (16kHz)
        if sample_rate != 16000:
            waveform = Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # 🔴 Ensure waveform is 1D (remove extra channel if stereo)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels

        label = self.annotations.iloc[idx]['label_consolidated_vocab']
        label = self.label2id[label]

        return waveform.squeeze(0), label


# Load the dataset
dataset = PodcastFillersDataset(csv_file='PodcastFillers_small.csv', audio_folder='audio/small_dataset_clips')

# Split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset.annotations, test_size=0.2, random_state=42)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

# Optional: Save preprocessed data to disk for further reuse (saving raw waveforms as .pt files)
def save_processed_data(dataset, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for idx in range(len(dataset)):
        waveform, label = dataset[idx]
        save_name = f"audio_{idx}.pt"
        save_filepath = os.path.join(save_path, save_name)
        
        # Save waveform and label as torch tensor
        torch.save((waveform, label), save_filepath)

# Save the processed data
save_processed_data(dataset, 'processed_data1')


