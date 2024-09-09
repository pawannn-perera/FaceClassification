import os
import keras
import streamlit as st 
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Celebrity Image Classification", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .header {
        font-size:40px; 
        color: Red;
        text-align:center;
    }
    .subheader {
        font-size:20px;
        color: #4B7BFF;
        text-align:center;
        font-family: 'Helvetica', sans-serif;
    }
    .image-container {
        display: flex;
        justify-content: center;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
model = load_model('new_model.keras')

# List of persons names for prediction
person_names = ['Angelina Jolie', 'Brad Pitt', 'Hugh Jackman', 'Johnny Depp', 'Leonardo DiCaprio']

def classify_images(image):
    # Open and preprocess the image for the model
    input_image = Image.open(image).resize((100, 100))
    input_image_array = np.array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    # Predict the class of the image
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])

    # Prepare results as a dictionary for all classes
    results_dict = {person_names[i]: round(float(result[i] * 100), 2) for i in range(len(person_names))}

    # Get the most likely class and format the result
    outcome = 'The Image belongs to "' + person_names[np.argmax(result)] + '" with a confidence score of "'+ str(np.max(result)*100)[:5] + '%"'
    return outcome, results_dict

# Page navigation

tab1, tab2 = st.tabs(["Home", "About"])

if page == "Home":
    # Home Page
    st.markdown('<h1 class="header">Hollywood Celebrity Image Classification</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p>The uploaded image should clearly show the face of the celebrity. For best results, make sure the image is well-lit and the face is clearly visible.</p>
    """, unsafe_allow_html=True)

    # Load and resize sample images
    sample_angelina = Image.open("./Sample/angelina_jolie.jpg").resize((250, 250))
    sample_brad = Image.open("./Sample/brad_pitt.jpg").resize((250, 250))
    sample_hugh = Image.open("./Sample/hugh_jackman.jpg").resize((250, 250))
    sample_johnny = Image.open("./Sample/johnny_depp.jpg").resize((250, 250))
    sample_leonardo = Image.open("./Sample/leonardo_dicaprio.jpg").resize((250, 250))

    # Display sample images
    st.markdown('<h2 class="subheader">Sample Images for use</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(sample_angelina, caption="Angelina Jolie", use_column_width=True)

    with col2:
        st.image(sample_brad, caption="Brad Pitt", use_column_width=True)

    with col3:
        st.image(sample_hugh, caption="Hugh Jackman", use_column_width=True)

    with col4:
        st.image(sample_johnny, caption="Johnny Depp", use_column_width=True)

    with col5:
        st.image(sample_leonardo, caption="Leonardo DiCaprio", use_column_width=True)

    # File uploader for image
    uploaded_file = st.file_uploader('Upload an Image')
    if uploaded_file is not None:
        # Display the image in the center
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(uploaded_file, width=250)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Classify the uploaded image
        outcome, results_dict = classify_images(uploaded_file)
        
        st.markdown('<h2 class="subheader">Classification Result:</h2>', unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: green;'>{outcome}</h3>", unsafe_allow_html=True)
        
        # Display the results as progress bars
        st.markdown('<h2 class="subheader">Prediction Scores:</h2>', unsafe_allow_html=True)
        for person, score in results_dict.items():
            st.markdown(f"**{person.capitalize()}**")
            st.progress(score / 100)

elif page == "About":
    # About Page
    st.markdown('<h1 class="header">About This App</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p>This application classify images of celebrities into one of the following five categories:</p>
        <ul>
            <li>Angelina Jolie</li>
            <li>Brad Pitt</li>
            <li>Hugh Jackman</li>
            <li>Johnny Depp</li>
            <li>Leonardo DiCaprio</li>
        </ul>
        <p>Upload an image of a celebrity, and the model will predict which celebrity the image belongs to, along with a confidence score. The app is designed to provide a fun and interactive way to see how well the model can recognize different celebrities.</p>
        <p>For best results, make sure the face in the image is clearly visible and well-lit. The model is trained to recognize these specific celebrities and may not perform as well with images of other people.</p>
    """, unsafe_allow_html=True)
