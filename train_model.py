import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
train_images = np.load("output/train_images.npy")
test_images = np.load("output/test_images.npy")
train_labels = np.load("output/train_labels.npy")
test_labels = np.load("output/test_labels.npy")

# Normalize pixel values to range [0,1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: real, AI-generated, artwork
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the trained model
model.save("output/ai_face_detector.h5")

print("Model training complete and saved as 'ai_face_detector.h5'")
