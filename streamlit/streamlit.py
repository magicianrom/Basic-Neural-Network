import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# Load model
network = load_model("pain_assessment_model.keras")

st.title("Facial Assessment")
st.write("Upload a facial image to classify the expression.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# Same class order as training
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image")

    # Prepare image
    image = image.resize((64, 64))
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict automatically
    prediction = network.predict(image_array, verbose=0)[0]
    predicted_class = np.argmax(prediction)

    st.subheader("Prediction")
    st.write(class_names[predicted_class])
    st.write(f"Confidence: {prediction[predicted_class] * 100:.2f}%")