import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Page configuration
st.set_page_config(
    page_title="Alphabet Recognition",
    page_icon="🔤",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Alphabet_Recognition_Model.keras")
    return model

model = load_model()

# Class labels (A-Z)
class_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Title
st.title("🔤 English Alphabet Recognition")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Drawing Canvas")
    
    # Create canvas
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=8,
        stroke_color="white",
        background_color="black",
        height=128,
        width=128,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    
    if st.button("🔮 Predict", use_container_width=True):
        if canvas_result.image_data is not None:
            # Get the image from canvas
            img = canvas_result.image_data
            
            # Convert to grayscale
            img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            
            # Resize to 28x28
            img_resized = cv2.resize(img_gray, (28, 28))
            
            # Normalize
            img_normalized = img_resized / 255.0
            
            # Reshape for model input
            img_input = img_normalized.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_input, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            predicted_letter = class_labels[predicted_class]
            
            # Display result
            st.markdown(f"### Predicted: **{predicted_letter}**")
            st.markdown(f"### Confidence: **{confidence:.1f}%**")
            
            # Show preprocessed image
            st.write("Preprocessed Image (28x28):")
            st.image(img_resized, width=100)
            
            # Show top 3 predictions
            st.write("---")
            st.write("Top 3 Predictions:")
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            for idx in top_3_indices:
                st.write(f"{class_labels[idx]}: {prediction[0][idx]*100:.1f}%")
        else:
            st.warning("Please draw something on the canvas first!")
    
    if st.button("🗑️ Clear Canvas", use_container_width=True):
        st.rerun()

# Instructions
st.markdown("---")
