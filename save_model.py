from keras.models import model_from_json

# Load model architecture
with open("emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)

# Load trained weights
emotion_model.load_weights("model/emotion_model.h5")

# Save the weights correctly
emotion_model.save_weights("emotion_model.weights.h5")  # Corrected file name

print("Weights saved successfully!")
emotion_model.save('emotion_model.keras')  # Saves full model
