import pickle
import tensorflow as tf
import numpy as np
import cv2
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import os

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load precomputed embeddings and filenames
if not os.path.exists("embeddings.pkl") or not os.path.exists("filenames.pkl"):
    raise ValueError("Missing 'embeddings.pkl' and 'filenames.pkl'. Run 'generate_embeddings.py' first.")

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load test image from 'Sample' folder
test_image_path = "Sample/sample.jpg"  # Change this to any image in Sample folder
if not os.path.exists(test_image_path):
    raise ValueError(f"Test image {test_image_path} not found!")

img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Find similar images
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])

# Show results
print("🖼️ Top 5 similar images found:")
for file in indices[0][1:6]:
    print(filenames[file])

# Display results
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('Similar Image', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)

cv2.destroyAllWindows()
