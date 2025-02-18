import streamlit as st
import os
import numpy as np
import glob
from zipfile import ZipFile
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine

# Extract zip file if not already extracted
def extract_zip(zip_file_path, extraction_directory):
    if not os.path.exists(extraction_directory):
        os.makedirs(extraction_directory)
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_directory)

# Load pre-trained model
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    return Model(inputs=base_model.input, outputs=base_model.output)

# Preprocess image for feature extraction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

# Extract image features
def extract_features(model, img_path):
    preprocessed_img = preprocess_image(img_path)
    features = model.predict(preprocessed_img)
    return features.flatten() / np.linalg.norm(features.flatten())

# Load image dataset and extract features
def load_image_data(image_directory, model):
    image_paths = [file for file in glob.glob(os.path.join(image_directory, '*.*'))
                   if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]
    all_features = [extract_features(model, img) for img in image_paths]
    return image_paths, all_features

# Recommend similar fashion items
def recommend_fashion_items(input_image_path, all_features, image_paths, model, top_n=5):
    input_features = extract_features(model, input_image_path)
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]
    return [image_paths[idx] for idx in similar_indices if image_paths[idx] != input_image_path]

# Streamlit UI
def main():
    st.title("Image-Based Fashion Recommendation System")

    zip_file_path = "women_fashion.zip"
    extraction_directory = "women_fashion"
    image_directory = os.path.join(extraction_directory, "women fashion")
    extract_zip(zip_file_path, extraction_directory)

    model = load_model()
    image_paths, all_features = load_image_data(image_directory, model)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])
    if uploaded_file is not None:
        file_path = os.path.join("temp.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        recommendations = recommend_fashion_items(file_path, all_features, image_paths, model)

        st.subheader("Recommended Fashion Items")
        for rec in recommendations:
            st.image(rec, use_column_width=True)

if __name__ == "__main__":
    main()

