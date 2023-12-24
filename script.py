import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Set your dataset directory
dataset_dir = 'archive'

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    return image[..., np.newaxis]  # Add channel dimension

# Function to read dataset csv file and preprocess the images and labels
def load_dataset(csv_path, image_folder):
    dataset = pd.read_csv(csv_path)
    images = []
    labels = []
    for img_path, label in zip(dataset['FILENAME'], dataset['IDENTITY']):
        img_full_path = os.path.join(image_folder, img_path)
        if os.path.exists(img_full_path):
            images.append(load_and_preprocess_image(img_full_path))
            labels.append(label)
        else:
            print(f"Image {img_full_path} does not exist.")
    return np.array(images), labels

# Load datasets
train_images, train_labels = load_dataset(os.path.join(dataset_dir, 'written_name_train_v2.csv'), os.path.join(dataset_dir, 'train_v2','train'))
val_images, val_labels = load_dataset(os.path.join(dataset_dir, 'written_name_validation_v2.csv'), os.path.join(dataset_dir, 'validation_v2','train'))
test_images, test_labels = load_dataset(os.path.join(dataset_dir, 'written_name_test_v2.csv'), os.path.join(dataset_dir, 'test_v2','train'))

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert labels to categorical
num_classes = len(label_encoder.classes_)
train_labels_categorical = to_categorical(train_labels_encoded, num_classes)
val_labels_categorical = to_categorical(val_labels_encoded, num_classes)
test_labels_categorical = to_categorical(test_labels_encoded, num_classes)

# Randomly sample 10% of the training data
np.random.seed(42)
subset_indices = np.random.choice(np.arange(len(train_images)), size=int(0.1 * len(train_images)), replace=False)

train_images_subset = train_images[subset_indices]
train_labels_categorical_subset = train_labels_categorical[subset_indices]

# Model architecture
input_shape = (128, 128, 1)
inputs = Input(shape=input_shape)

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Reshape for RNN layers
new_shape = ((128 // (2**3)) * (128 // (2**3)) * 128,)
x = Reshape(target_shape=new_shape)(x)

# RNN layers
x = LSTM(128, return_sequences=True)(x)
x = Dropout(0.25)(x)
x = LSTM(64, return_sequences=False)(x)

# Output layer
outputs = Dense(num_classes, activation='softmax')(x)

# Compile the model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the subset
history = model.fit(
    train_images_subset, train_labels_categorical_subset,
    validation_data=(val_images, val_labels_categorical),
    epochs=10,
    batch_size=16
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels_categorical)
print(f"Test accuracy: {test_accuracy}")

# Save model
model.save('handwritten_text_recognition_model.h5')