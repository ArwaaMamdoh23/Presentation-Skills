from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ✅ Load full model (architecture + weights)
emotion_model = load_model("emotion_model.keras")  
print("✅ Model loaded successfully!")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
        r'D:\GitHub\Presentation-Skills\DataSet\test',  
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Make predictions on test data
predictions = emotion_model.predict(test_generator)

# Confusion matrix
print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=[
    "Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised", "Other"
])
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))

# Evaluate the model on the test data
test_loss, test_acc = emotion_model.evaluate(test_generator)
print(f"📊 Test Accuracy: {test_acc * 100:.2f}%")

