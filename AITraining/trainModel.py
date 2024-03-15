import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Directory paths
train_dir = 'set1'

# Model parameters
img_width, img_height = 256, 256  # Must match the preprocessing size
batch_size = 32
epochs = 40
num_classes = 4  # Adjust based on the number of directions including "arrived"

# Data preparation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    validation_split=0.2  # Splitting data into 80% training and 20% validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # For multi-class classification
    subset='validation'  # Set as validation data
)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('wizard101_navigation_model.keras')
print("Model training completed and saved as 'wizard101_navigation_model.keras'")
