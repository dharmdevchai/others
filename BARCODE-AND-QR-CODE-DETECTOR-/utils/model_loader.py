import os
import torch
import streamlit as st

@st.cache_resource
def load_pytorch_model(path):
    """Load PyTorch model weights safely with error handling."""
    if not os.path.exists(path):
        return None, f"Model file not found at {path}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(path, map_location=device)
        return state_dict, "Model loaded successfully!"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"
