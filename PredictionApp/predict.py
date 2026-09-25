import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Disable oneDNN optimizations to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Image dimensions
imgWidth = 256
imgHeight = 256

# Define the weather classes
classes = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

# Load the pre-trained model
model_path = "C:\\Users\\Shreyas\\Desktop\\Weather_ANN\\bestWeatherModel.h5"
model = load_model(model_path)

# Compile the model to avoid optimizer warnings
model.compile(optimizer='Adam')

# Function to preprocess the image
def prepareImage(image_path):
    image = load_img(image_path, target_size=(imgHeight, imgWidth))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)  # Add batch dimension
    imgResult = imgResult / 255.0  # Normalize the image
    return imgResult

def main():
    if len(sys.argv) < 2:
        print("Please provide an image path")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        sys.exit(1)

    try:
        # Preprocess the image
        print(f"Preparing image: {image_path}")
        image = prepareImage(image_path)

        # Run prediction
        print("Running prediction...")
        result = model.predict(image)
        predicted_class = np.argmax(result, axis=1)

        # Print the prediction result
        print(f"Predicted weather class: {classes[predicted_class[0]]}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
