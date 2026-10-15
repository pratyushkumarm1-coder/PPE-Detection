import streamlit as st
import requests

st.set_page_config(page_title="Helmet Detection", page_icon="🪖", layout="centered")

st.title("🪖 Helmet Detection Web App")
st.write("Upload an image or use your webcam to detect if workers are wearing helmets!")

# Use the FastAPI server URL
API_URL = "http://localhost:8000/detect/annotated"

option = st.radio("Choose input method:", ("Webcam", "Upload Image"))

uploaded_file = None
if option == "Webcam":
    uploaded_file = st.camera_input("Take a picture with your webcam")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image if it's not the webcam (webcam auto-displays)
    if option == "Upload Image":
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("🔍 Detect Helmets", type="primary"):
        with st.spinner("Analyzing image for helmets..."):
            try:
                # Send the file to the FastAPI backend
                # Ensure the filename has an extension so the API mime check passes
                filename = uploaded_file.name if hasattr(uploaded_file, 'name') and uploaded_file.name else 'webcam.jpg'
                files = {"file": (filename, uploaded_file.getvalue(), "image/jpeg")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    st.success("Analysis Complete!")
                    st.image(response.content, caption="Detected Helmets", use_container_width=True)
                else:
                    st.error(f"Failed to analyze image. API returned status code: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure `uvicorn main:app --reload` is running in another terminal!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
