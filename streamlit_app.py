import streamlit as st
import requests

st.title("📄 Invoice Data Parser")

uploaded_file = st.file_uploader("Upload Invoice", type=["jpg", "png", "jpeg"])

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post(
        "http://127.0.0.1:8000/extract-invoice", files=files
    )

    if response.status_code == 200:
        st.success("Extraction Successful")
        st.json(response.json())
    else:
        st.error("Extraction Failed")
