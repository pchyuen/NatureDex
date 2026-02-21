import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===================================
# Page config and styling
# ===================================
st.set_page_config(page_title="NatureDex: Animal Identification", page_icon="🐾", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .ranger-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 8px solid #2E7D32;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    h1 { color: #1B5E20; text-align: center; }
    .stProgress > div > div > div > div { background-color: #2E7D32; }
    </style>
    """, unsafe_allow_html=True)

st.title("NatureDex")
st.markdown("<h4 style='text-align: center; color: #666;'>Field Guide and Species Identification!!</h4>", unsafe_allow_html=True)
st.divider()

# ===================================
# Animal's Ranger Cards
# ===================================
RANGER_DATA = {
    "butterfly": {"habitat": "Gardens, Meadows & Tropical Forests", "diet": "Nectar from flowers", "fact": "Butterflies taste their food with their feet to find where to lay eggs."},
    "cat": {"habitat": "Urban & Domestic environments", "diet": "Carnivore", "fact": "Cats have a specialised collarbone that allows them to always land on their feet."},
    "chicken": {"habitat": "Farms & Scrublands", "diet": "Omnivore (Seeds, Insects)", "fact": "Chickens can recognise and remember over 100 different faces of people and animals."},
    "cow": {"habitat": "Pastures, Grasslands & Farms", "diet": "Herbivore (Grass, Hay)", "fact": "Cows have a nearly 360-degree panoramic vision to spot predators from any direction."},
    "dog": {"habitat": "Domestic & Other environments", "diet": "Omnivore", "fact": "A dog's sense of smell is up to 100,000 times more acute than a human's."},
    "elephant": {"habitat": "African Savannahs & Asian Forests", "diet": "Herbivore (Roots, Grass, Bark)", "fact": "Elephants communicate over long distances using low-frequency sounds that travel through the ground."},
    "horse": {"habitat": "Open Grasslands & Farms", "diet": "Herbivore (Hay, Grains)", "fact": "Horses have the largest eyes of any land mammal and can sleep while standing up."},
    "sheep": {"habitat": "Highlands, Pastures & Mountains", "diet": "Herbivore (grass)", "fact": "Sheep have rectangular pupils that provide a 320-degree field of vision."},
    "spider": {"habitat": "Global (except Antarctica)", "diet": "Carnivore (mostly insects)", "fact": "Spider silk is five times stronger than steel of the same diameter."},
    "squirrel": {"habitat": "Woodlands & Urban Parks", "diet": "Omnivore (Nuts, Seeds, Fruit)", "fact": "Squirrels plant thousands of trees every year by forgetting where they buried their nuts."}
}

# ===================================
# Load model
# ===================================
@st.cache_resource
def load_naturedex():
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(150, 150, 3), 
        include_top=False, 
        weights=None
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.255),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    try:
        model.load_weights('naturedex_best_model.keras')
    except:
        temp_model = tf.keras.models.load_model('naturedex_best_model.keras', compile=False)
        model.set_weights(temp_model.get_weights())
    
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_naturedex()
    st.sidebar.success("Model Loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# ===================================
# Prediction
# ===================================

uploaded_file = st.file_uploader("Upload an animal photo for identification...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")
    image = Image.open(uploaded_file)
    
    with col1:
        st.markdown("### Captured Subject")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### Species Analysis")
        with st.spinner("Analysing features..."):
            
            # Preprocessing
            img = image.convert('RGB').resize((150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction logic
            predictions = model.predict(img_array)
            probs = tf.nn.softmax(predictions[0])
            result_index = np.argmax(probs)
            result_class = class_names[result_index]
            confidence = 100 * np.max(probs)

            # Result
            st.markdown(f"**Species Identified:** {result_class.capitalize()}")
            st.progress(int(confidence))
            st.write(f"Confidence Score: **{confidence:.2f}%**")

            # RANGER CARD
            if result_class in RANGER_DATA:
                card = RANGER_DATA[result_class]
                st.markdown("---")
                st.markdown(f"""
                <div class="ranger-card">
                    <h3 style='margin-top:0;'>Ranger Field Notes: {result_class.capitalize()}</h3>
                    <p><b>Natural Habitat:</b> {card['habitat']}</p>
                    <p><b>Primary Diet:</b> {card['diet']}</p>
                    <hr>
                    <p><b>Did you know?</b> {card['fact']}</p>
                </div>
                """, unsafe_allow_html=True)

st.divider()
st.caption("NatureDex | A DLOR project 2026")
