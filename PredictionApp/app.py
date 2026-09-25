import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tempfile

# Disable oneDNN optimizations to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Image dimensions
imgWidth = 256
imgHeight = 256

# Define the weather classes
classes = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

# Load the pre-trained model once when the app starts
model_path = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\bestWeatherModel.h5"
model = load_model(model_path)

# Compile the model to avoid optimizer warnings
model.compile(optimizer='adam')

# Function to preprocess the image for prediction
def prepare_image(image_path):
    image = load_img(image_path, target_size=(imgHeight, imgWidth))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)  # Add batch dimension
    img_result = img_result / 255.0  # Normalize the image
    return img_result

# Function to get modern gradient backgrounds based on weather prediction
def get_modern_gradient(weather):
    gradients = {
        'cloudy': 'linear-gradient(to right, #1C1C1C, #434343)',  # Dark cloudy feel
        'foggy': 'linear-gradient(to right, #D3CBB8, #E0D5C6)',   # Soft foggy tones
        'rainy': 'linear-gradient(to right, #00C6FF, #0072FF)',   # Vibrant blue rain
        'shine': 'linear-gradient(to right, #FDC830, #F37335)',   # Bright sunny feel
        'sunrise': 'linear-gradient(to right, #FF5F6D, #FFC371)'  # Warm sunrise colors
    }
    return gradients.get(weather, 'linear-gradient(to right, #4e54c8, #8f94fb)')  # Default fallback

# Function to update the CSS background dynamically
def apply_background_css(gradient):
    st.markdown(f"""
    <style>
        .reportview-container {{
            background: {gradient};
        }}
    </style>
    """, unsafe_allow_html=True)

# Streamlit app setup
st.set_page_config(page_title="Sky Weather Predictor", layout="wide")

# Initialize the background gradient in session state if not present
if 'background' not in st.session_state:
    st.session_state.background = get_modern_gradient('default')

# Apply the initial background gradient
apply_background_css(st.session_state.background)

# Custom CSS for the app layout and design
st.markdown("""
<style>
    .big-font {
        font-size:50px !important;
        font-weight:bold;
        color: white;
        text-align: center;
    }
    .sub-font {
        font-size:30px !important;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Title and instructions
st.markdown('<p class="big-font">Sky Weather Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-font">Upload an image of the sky and get a weather prediction!</p>', unsafe_allow_html=True)

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Temporarily save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict Weather'):
        with st.spinner('Predicting...'):
            # Preprocess the image and make prediction
            try:
                image = prepare_image(tmp_file_path)
                result = model.predict(image)
                predicted_class = np.argmax(result, axis=1)
                prediction = classes[predicted_class[0]]

                # Update the UI with the prediction result
                st.success(f'Predicted Weather: {prediction.capitalize()}')
                st.balloons()

                # Change background based on prediction
                st.session_state.background = get_modern_gradient(prediction)

                # Apply the new background immediately
                apply_background_css(st.session_state.background)

            except Exception as e:
                st.error(f"Failed to make a prediction: {e}")

    # Clean up the temporary file after use
    os.unlink(tmp_file_path)

# How the app works section
st.markdown("""
    <div style='text-align: center; color: white; margin-top: 50px;'>
        <h3>How it works</h3>
        <p>1. Upload an image of the sky using the file uploader above.</p>
        <p>2. Click the 'Predict Weather' button.</p>
        <p>3. Our AI model will analyze the image and predict the weather condition!</p>
        <p>4. Watch as the background changes to match the predicted weather with a modern gradient!</p>
    </div>
""", unsafe_allow_html=True)
