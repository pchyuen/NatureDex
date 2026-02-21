import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===================================
# page config
# ===================================

st.set_page_config(page_title="NatureDex Animal Classifier", page_icon="🐾")
st.title("NatureDex: Animal Identification")
st.markdown("Upload a photo to identify the species (butterfly, cat, dog, chicken, cow, elephant, horse, sheep, spider, squirrel).")

# ===================================
# loading of models and labells
# ===================================

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('naturedex_best_model.keras')
    with open('class_names.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, labels = load_resources()

# ===================================
# uploader for images
# ===================================

file = st.file_uploader("Upload an animal photo...", type=["jpg", "png", "jpeg"])

if file:
  # ===================================
    # Display image
  # ===================================
  
    img = Image.open(file)
    st.image(img, caption="Uploaded Photo", use_column_width=True)
    
    # ===================================
    # Preprocessing
    # ===================================

    st.write("Processing...")
    processed_img = img.convert('RGB').resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(processed_img)
    img_array = np.expand_dims(img_array, axis=0) # adds batch dimension

# ===================================
    # Prediction
# ===================================

    preds = model.predict(img_array)
    score = tf.nn.softmax(preds[0]) # Get confidence levels
    
    species = labels[np.argmax(score)]
    confidence = 100 * np.max(score)

# ===================================
    # Results
# ===================================

    st.success(f"Identification Result: **{species.upper()}**")
    st.info(f"AI Confidence Score: **{confidence:.2f}%**")
