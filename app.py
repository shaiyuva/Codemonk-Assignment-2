import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from tqdm import tqdm

# Load Pre-trained ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Load images and extract features
image_folder = "images"
filenames = []
feature_list = []

for file in tqdm(os.listdir(image_folder)):
    img_path = os.path.join(image_folder, file)
    if file.endswith(('jpg', 'jpeg', 'png')):  # Ensure only image files are processed
        filenames.append(img_path)
        feature_list.append(extract_features(img_path, model))

# Save extracted features and filenames
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ Embeddings and filenames saved successfully!")
