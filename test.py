import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define paths to the saved model and the test data
model_path = 'flower_classifier_model.h5'
test_data_dir = 'dataset/val'

# Load the saved model
model = load_model(model_path)

# Get the class labels
class_labels = sorted(os.listdir(os.path.join(test_data_dir)))

# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0

# Function to preprocess images for prediction
def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to perform predictions on a single image
def predict_single_image(image_path):
    global total_images, correct_predictions
    total_images += 1
    actual_class = os.path.basename(os.path.dirname(image_path))
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    if actual_class == predicted_class:
        correct_predictions += 1
    return predicted_class, confidence, actual_class

# Test the model on the test data
for class_label in class_labels:
    class_dir = os.path.join(test_data_dir, class_label)
    try:
        for item in os.listdir(class_dir):
            image_path = os.path.join(class_dir, item)
            if os.path.isfile(image_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')):
                predicted_class, confidence, actual_class = predict_single_image(image_path)
                print(f"Image: {item}, Predicted Class: {predicted_class}, Actual Class: {actual_class}, Confidence: {confidence}")
    except NotADirectoryError:
        pass

# Calculate accuracy
accuracy = correct_predictions / total_images
print(f"Total Images: {total_images}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy * 100:.2f}%")
