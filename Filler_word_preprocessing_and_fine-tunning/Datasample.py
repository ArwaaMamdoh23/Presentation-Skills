import os
import pandas as pd
import shutil

base_path = "D:/PodcastFillers/audio/clip_wav" 
splits = ["train", "test", "validation", "extra"]

split_files = {}
for split in splits:
    split_path = os.path.join(base_path, split)
    split_files[split] = set(os.listdir(split_path)) if os.path.exists(split_path) else set()

df = pd.read_csv("D:/PodcastFillers/metadata/PodcastFillers.csv")
df["split"] = df["clip_name"].apply(lambda clip: next((s for s in splits if clip in split_files[s]), "unknown"))

sample_sizes = {"train": 10000, "test": 2000, "validation": 2000, "extra": 1000}
df_sampled = pd.concat([
    df[df["split"] == split].sample(n=min(sample_sizes[split], len(df[df["split"] == split])), random_state=42)
    for split in splits
])
df_sampled.to_csv("PodcastFillers_small.csv", index=False)

destination_folder = "audio/small_dataset_clips"
os.makedirs(destination_folder, exist_ok=True)
for _, row in df_sampled.iterrows():
    src_path = os.path.join(base_path, row["split"], row["clip_name"])
    dest_path = os.path.join(destination_folder, row["clip_name"])
    if os.path.exists(src_path):
        shutil.copy(src_path, dest_path)

print(f"Sampled dataset saved with {df_sampled.shape[0]} rows and audio files copied to {destination_folder}.")

