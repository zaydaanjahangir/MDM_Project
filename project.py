import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants
image_width = 256
image_height = 256
batch_size = 64
epochs = 50
num_classes = 14

# Define paths to the training and validation data
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'

# Calculate steps per epoch
class_samples = {}
for class_name in os.listdir(train_data_dir):
    class_dir = os.path.join(train_data_dir, class_name)
    if os.path.isdir(class_dir):
        jpg_files = [file for file in os.listdir(class_dir) if file.lower().endswith('.jpg')]
        class_samples[class_name] = len(jpg_files)

train_samples = sum(class_samples.values())
steps_per_epoch = train_samples // batch_size

# print(class_samples)
# {'iris': 1041, 'astilbe': 726, 'coreopsis': 1035, 'calendula': 1011, 'rose': 986, 'bellflower': 872,
# 'common_daisy': 978, 'sunflower': 1013, 'water_lily': 977, 'california_poppy': 1021, 'carnation': 924,
# 'black_eyed_susan': 986, 'dandelion': 1038, 'tulip': 1034}
# Found 13642 images belonging to 14 classes.
# Found 98 images belonging to 14 classes.


# Data preprocessing and augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=42,
    save_format='jpg')

# Data preprocessing for validation data
val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42,
    save_format='jpg')

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size)

# Save the trained model
model.save('flower_classifier_model.h5')
