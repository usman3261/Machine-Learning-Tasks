import streamlit as st
from transformers import pipeline
import os

st.set_page_config(page_title="News Classifier")
st.title("📰 News Topic Classifier")

# Use an absolute path to avoid confusion
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "news_model")

# --- DEBUGGING SECTION ---
# This will show you on the web page if the folder is actually visible
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Folder NOT found at: {MODEL_PATH}")
    st.info(f"Existing items in this folder: {os.listdir(current_dir)}")
else:
    st.success("✅ Model folder detected!")
# -------------------------

@st.cache_resource
def load_pipeline():
    # If the folder exists and has files inside it
    if os.path.exists(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
        return pipeline("text-classification", model=MODEL_PATH, device=-1)
    return None

classifier = load_pipeline()

if classifier:
    label_map = {"LABEL_0": "World", "LABEL_1": "Sports", "LABEL_2": "Business", "LABEL_3": "Sci/Tech"}
    
    text = st.text_area("Enter News Headline:", "The tech company released a new smartphone...")
    
    if st.button("Classify"):
        pred = classifier(text)[0]
        label = label_map.get(pred['label'], pred['label'])
        st.success(f"Category: {label} (Confidence: {pred['score']:.2%})")