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
class_labels = sorted(
    label for label in os.listdir(os.path.join(test_data_dir)) if not label.startswith('classname.txt'))
print(class_labels)


# Function to preprocess images for prediction
def preprocess_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to perform predictions on a single image
def predict_single_image(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence


# Initialize variables for accuracy calculation
total_images = 0
correct_predictions = 0

# Test the model on the test data
for class_label in class_labels:
    class_dir = os.path.join(test_data_dir, class_label)
    # Ensure that the item is a directory
    if os.path.isdir(class_dir):
        for item in os.listdir(class_dir):
            image_path = os.path.join(class_dir, item)
            if os.path.isfile(image_path) and item.lower().endswith(('.png', '.jpg', '.jpeg')) and not item.startswith(
                    'classname.txt'):
                predicted_class, confidence = predict_single_image(image_path)
                actual_class = class_label
                total_images += 1
                if actual_class == predicted_class:
                    correct_predictions += 1
                print(
                    f"Image: {item}, Predicted Class: {predicted_class}, Actual Class: {actual_class}, Confidence: {confidence :.3f}")

# Calculate accuracy
accuracy = correct_predictions / total_images
print(f"Total Images: {total_images}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy * 100:.2f}%")

# Total Images: 98, Correct Predictions: 63, Accuracy: 64.29%
