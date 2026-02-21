import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===================================
# Page configuration
# ===================================
st.set_page_config(page_title="NatureDex: Animal Identification", page_icon="🐾")
st.title("NatureDex")
st.markdown("Upload a photo to identify the species (butterfly, cat, dog, chicken, cow, elephant, horse, sheep, spider, squirrel).")

# ===================================
# Load model and class names
# ===================================

@st.cache_resource
def load_naturedex():
    model = tf.keras.models.load_model('naturedex_best_model.keras', compile=False)
    
    # Load labels
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_naturedex()
    st.sidebar.success("Model loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ===================================
# upload and process image
# ===================================

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the user image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("🔍 Analyzing species...")
    
    # ===================================
    # Preprocessing
    # Converts to RGB and resize to training size
    # ===================================

    img = image.convert('RGB').resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # adds batch dimension

    # ===================================
    # Handling tensors
    # ===================================
    predictions = model.predict(img_array)
    
    if isinstance(predictions, list):
        predictions = predictions[0]
    probs = tf.nn.softmax(predictions[0])
    
    result_index = np.argmax(probs)
    result_class = class_names[result_index]
    confidence = 100 * np.max(probs)

    # ===================================
    # Results
    # ===================================
    st.header(f"Result: {result_class.capitalize()}")
    
    # ===================================
    # Progress bar
    # ===================================
    
    st.progress(int(confidence))
    st.info(f"AI Confidence Score: {confidence:.2f}%")
    
    if confidence < 60:
        st.warning("Low confidence: Try a clearer photo or a different angle.")