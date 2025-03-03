# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15G3UIkEK_EE--1U_ldMctdLh6097d-zn
"""

# from google.colab import drive
# drive.mount('/content/drive')

# %%writefile app.py
import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load saved features and filenames
image_features_path = "/content/drive/MyDrive/Images_features.pkl"
filenames_path = "/content/drive/MyDrive/filenames.pkl"

with open(image_features_path, "rb") as f:
    image_features = pkl.load(f)

with open(filenames_path, "rb") as f:
    filenames = pkl.load(f)

# Load pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.models.Sequential([model, tf.keras.layers.GlobalMaxPool2D()])

# Build Nearest Neighbors Model
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(image_features)

def extract_features(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    return result / norm(result)

def recommend_similar_images(uploaded_img):
    feature_vector = extract_features(uploaded_img)
    distances, indices = neighbors.kneighbors([feature_vector])
    return [filenames[i] for i in indices[0]]

# Streamlit UI
st.title("Fashion Recommendation System")
st.write("Upload an image to find visually similar fashion items.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = PILImage.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Finding recommendations...")
    recommendations = recommend_similar_images(image)

    st.write("## Recommended Items:")
    cols = st.columns(len(recommendations))
    for col, img_path in zip(cols, recommendations):
        col.image(img_path, use_column_width=True)

