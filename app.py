import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===================================
# Page config
# ===================================
st.set_page_config(page_title="NatureDex: Animal Identification", page_icon="🐾")
st.title("NatureDex")
st.markdown("Identify whether or not it's one of these 10 animals! cat, dog, butterfly, spider, cow, squirrel, chicken, elephant, sheep, horse")

# ===================================
# Load model
# ===================================
@st.cache_resource
def load_naturedex():
    # rebuilds the exact architecture from training to avoid loading errors
    # will load the custom weights in the next step
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(150, 150, 3), 
        include_top=False, 
        weights=None
    )
    
    # custom layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.255), # best dropout from Optuna
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # weights from your saved file
    try:
        model.load_weights('naturedex_best_model.keras')
    except:
        temp_model = tf.keras.models.load_model('naturedex_best_model.keras', compile=False)
        model.set_weights(temp_model.get_weights())
    
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

# ===================================
# Image upload
# ===================================
uploaded_file = st.file_uploader("Upload an animal photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analysing...")
    
    # Preprocessing
    img = image.convert('RGB').resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ===================================
    # Prediction handling
    # ===================================
    predictions = model.predict(img_array)
    probs = tf.nn.softmax(predictions[0])
    result_index = np.argmax(probs)
    result_class = class_names[result_index]
    confidence = 100 * np.max(probs)

    # ===================================
    # Results
    # ===================================
    st.header(f"Result: {result_class.capitalize()}")
    st.progress(int(confidence))
    st.info(f"Confidence Score: {confidence:.2f}%")

