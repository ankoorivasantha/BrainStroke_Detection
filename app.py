import streamlit as st
import joblib
import numpy as np
from PIL import Image
import base64

# Load the brain stroke detection model
model = joblib.load('models/mobilenetmodel.pkl')

# Define the image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")  # Ensure the image is RGB
    image = image.resize(target_size)  # Resize to target input shape
    image = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image = image.reshape(1, *target_size, 3)  # Reshape to match model input (batch_size, height, width, channels)
    return image

# Function to encode image to Base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get the Base64 string of the local image
image_path = "C:\\Users\\Vasantha\\OneDrive\\Desktop\\wallpaper.jpg"  # Replace with the path to your local image
background_image = get_base64_of_image(image_path)

# Add custom CSS for background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{background_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.8);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app setup
st.title("Edge Computing Based Smart Health Care for Detecting Brain Strokes")
st.markdown("Upload a CT scan image to detect the presence of brain stroke.")

# Upload CT scan image
uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CT Scan', use_container_width=True)

    # Preprocess the image for model input
    processed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(processed_image)

    # Display the result based on prediction output
    if isinstance(prediction[0], np.ndarray):  # If output is probability
        prediction = prediction[0][0] > 0.5  # Threshold at 0.5
    else:
        prediction = prediction[0]  # Assuming binary output 0 or 1

    if prediction == 1:
        st.error("Brain Stroke Detected")
    else:
        st.success("No Stroke Detected")
