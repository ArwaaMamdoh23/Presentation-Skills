import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from tqdm import tqdm  # For progress bar

# Load the preprocessed dataset
X_train, y_train = torch.load("train_data.pt")
X_val, y_val = torch.load("val_data.pt")

# Create PyTorch Datasets & DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load Wav2Vec2 pre-trained model (for classification)
num_labels = 6  # Total labels: {'Uh': 0, 'Words': 1, 'Laughter': 2, 'Um': 3, 'Music': 4, 'Breath': 5}
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=num_labels)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Move model to CPU (since you don't have a GPU)
device = torch.device("cpu")
model.to(device)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    
    accuracy = correct / len(val_loader.dataset)
    return total_loss / len(val_loader), accuracy

# Training loop
num_epochs = 5  # Adjust as needed
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")

    # Save checkpoint after each epoch
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

# Save the final fine-tuned model
torch.save(model.state_dict(), "fine_tuned_wav2vec2.pth")
print("\n✅ Model training complete and saved!")
