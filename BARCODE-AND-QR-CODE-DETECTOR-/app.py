import numpy as np
import streamlit as st
from PIL import Image
from utils.model_loader import load_pytorch_model
from utils.processor import process_image

# Page Configuration
st.set_page_config(
    page_title="Barcode & QR Code Detector",
    page_icon="📦",
    layout="wide"
)

# Sidebar for Model Settings
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose Detector Model",
    (
        "Barcode Detector (barcode_detector.pth)",
        "Scratch Detector (barcode_qr_detector_scratch.pth)"
    )
)

# Map selection to file path
model_path = "models/barcode_detector.pth" if "barcode_detector.pth" in model_choice else "models/barcode_qr_detector_scratch.pth"

# Load Model Status via Utility Module
model_weights, load_message = load_pytorch_model(model_path)
if model_weights is not None:
    st.sidebar.success(load_message)
else:
    st.sidebar.error(load_message)

# Main UI Interface
st.markdown("<h1>📦 Barcode & QR Code Detector</h1>", unsafe_allow_html=True)
st.write("Upload an image containing barcodes or QR codes to detect, classify, and decode them using your custom PyTorch models and pyzbar pipeline.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image via PIL
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    
    # Process image using modular backend function
    annotated_image, results = process_image(image_np)

    # Layout: Side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_pil, use_container_width=True)
        
    with col2:
        st.subheader("Detection Results")
        st.image(annotated_image, use_container_width=True)

    st.markdown("---")
    st.subheader("Decoded Content & Classification:")
    
    if results:
        for idx, res in enumerate(results, 1):
            category = "QR Code" if "QR" in res['type'].upper() else "Barcode"
            
            st.info(
                f"**Item {idx}**\n\n"
                f"- **Detected Category:** {category}\n"
                f"- **Specific Type:** {res['type']}\n"
                f"- **Decoded Data:** `{res['data']}`"
            )
    else:
        st.warning("No scannable text extracted automatically. Model pipeline active, but code features may require explicit ROI cropping or threshold adjustments.")
