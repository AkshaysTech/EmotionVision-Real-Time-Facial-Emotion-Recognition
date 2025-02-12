import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

# Function to load and preprocess images
def load_and_preprocess_data(data_dir, image_size):
    X = []
    y = []
    
    emotions = ['happy', 'sad', 'angry']
    label_encoder = LabelEncoder()
    label_encoder.fit(emotions)

    for emotion in emotions:
        emotion_dir = os.path.join(data_dir, emotion)
        for image_file in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, image_file)
            image = load_img(image_path, target_size=image_size)
            image_array = img_to_array(image) / 255.0  # Normalize pixel values
            X.append(image_array)
            y.append(emotion)

    X = np.array(X)
    y = label_encoder.transform(y)
    y = to_categorical(y, num_classes=len(emotions))
    
    return X, y

# Load and preprocess data
data_dir = "data"
image_size = (128, 128)
X, y = load_and_preprocess_data(data_dir, image_size)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create CNN model
input_shape = X_train[0].shape
num_classes = len(y_train[0])
model = create_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint_callback = callbacks.ModelCheckpoint('emotion_detection_model.h5', save_best_only=True, verbose=1)
early_stopping_callback = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback, early_stopping_callback])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('emotion_detection_model.h5')
