import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load embeddings and filenames
if not os.path.exists("embeddings.pkl") or not os.path.exists("filenames.pkl"):
    st.error("Missing 'embeddings.pkl' and 'filenames.pkl'. Run 'generate_embeddings.py' first.")
    st.stop()

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('🛍️ Fashion Recommender System')

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except:
        return None

# Feature extraction
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation function
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Upload image
uploaded_file = st.file_uploader("📂 Upload an Image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        display_image = Image.open(file_path)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Extract features
        features = extract_features(file_path, model)

        # Get recommendations
        indices = recommend(features, feature_list)

        # Display recommended images
        st.subheader("💡 Recommended Fashion Items")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
