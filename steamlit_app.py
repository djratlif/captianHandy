import streamlit as st
from PIL import Image
from image_utils import get_image_paths, embed_images, build_faiss_index, embed_single_image
import faiss
import numpy as np
import os

st.set_page_config(page_title="CLIP Image Search", layout="wide")
st.title("🔍 CLIP Image Similarity Search")

@st.cache_resource
def setup_index():
    image_paths = get_image_paths()
    embeddings = embed_images(image_paths)
    index = build_faiss_index(embeddings)
    return index, image_paths

index, image_paths = setup_index()

uploaded_file = st.file_uploader("Upload an image to search for similar ones:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", use_column_width=True)

    query_embed = embed_single_image(query_image)
    D, I = index.search(query_embed, k=min(5, len(image_paths)))

    st.subheader("Top Similar Images")
    cols = st.columns(len(I[0]))
    for col, idx in zip(cols, I[0]):
        similar_img = Image.open(image_paths[idx])
        col.image(similar_img, caption=os.path.basename(image_paths[idx]), use_column_width=True)
