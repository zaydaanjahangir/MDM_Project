import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16  # Import VGG16 model class

# Define constants
image_width = 256
image_height = 256
batch_size = 32
epochs = 20
num_classes = 14

# Define paths to the training and validation data
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'

# Load pre-trained VGG16 model (excluding top layers)
pretrained_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model = VGG16(weights=pretrained_weights_path, include_top=False, input_shape=(image_width, image_height, 3))

# Build the model on top of the pre-trained VGG16
model = Sequential()
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data preprocessing and augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

# Data preprocessing for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size)

# Save the trained model
model.save('flower_classifier_model_transfer_learning.h5')
