import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from tqdm import tqdm

# Load Pretrained Model (ResNet50)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Add a pooling layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Load and resize image
    img_array = image.img_to_array(img)  # Convert to array
    expanded_img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    preprocessed_img = preprocess_input(expanded_img_array)  # Preprocess for ResNet
    result = model.predict(preprocessed_img).flatten()  # Extract features
    normalized_result = result / norm(result)  # Normalize features
    return normalized_result

# Define the path to your image dataset
image_folder = 'images'  # Make sure all images are inside this folder
if not os.path.exists(image_folder):
    raise ValueError(f"Image folder '{image_folder}' not found. Please check the path.")

# Get all image filenames
filenames = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Extract and store embeddings
feature_list = []

print("Extracting features from images...")

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save embeddings and filenames as pickle files
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('filenames.pkl', 'wb') as f:
    pickle.dump(filenames, f)

print("Feature extraction completed. 'embeddings.pkl' and 'filenames.pkl' saved successfully!")
