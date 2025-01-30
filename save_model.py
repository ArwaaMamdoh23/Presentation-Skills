from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model architecture
with open("emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)

# Load trained weights
emotion_model.load_weights("model/emotion_model.h5")
print("✅ Model loaded successfully!")

# Initialize image data generator for training data
train_data_gen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_generator = train_data_gen.flow_from_directory(
    r'D:\GitHub\Presentation-Skills\DataSet\train',  # Path to training dataset
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)

# Evaluate model on training data
train_loss, train_acc = emotion_model.evaluate(train_generator, steps=len(train_generator))
print(f"📊 Estimated Training Accuracy: {train_acc * 100:.2f}%")
