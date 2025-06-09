import streamlit as st
import os
import tempfile
from inference import main as wav2lip_inference
import cv2
import numpy as np
from PIL import Image
import torch
import traceback

st.set_page_config(page_title="Wav2Lip - Lip Sync Generator", layout="wide")

st.title("Wav2Lip - Lip Sync Generator")
st.write("Upload an image and audio file to generate a lip-synced video.")

# Create a directory for temporary files if it doesn't exist
os.makedirs("temp", exist_ok=True)

# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        # Save the uploaded image
        image_path = os.path.join("temp", "input_image.jpg")
        with open(image_path, "wb") as f:
            f.write(image_file.getvalue())
        st.image(image_file, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    if audio_file is not None:
        # Save the uploaded audio
        audio_path = os.path.join("temp", "input_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_file.getvalue())
        st.audio(audio_file)

# Process button
if st.button("Generate Lip Sync Video"):
    if image_file is None or audio_file is None:
        st.error("Please upload both an image and an audio file.")
    else:
        with st.spinner("Generating lip sync video... This may take a few minutes."):
            try:
                # Set up the arguments for the inference
                args = type('Args', (), {
                    'checkpoint_path': 'checkpoints/wav2lip.pth',
                    'face': image_path,
                    'audio': audio_path,
                    'outfile': os.path.join("temp", "output.mp4"),
                    'static': True,
                    'fps': 25,
                    'pads': [0, 20, 0, 0],
                    'face_det_batch_size': 16,
                    'wav2lip_batch_size': 128,
                    'resize_factor': 1,
                    'crop': [0, -1, 0, -1],
                    'box': [-1, -1, -1, -1],
                    'rotate': False,
                    'nosmooth': False,
                    'noresize': False,
                    'img_size': 96
                })()

                # Run the inference
                wav2lip_inference(args)

                # Display the result
                st.success("Video generated successfully!")
                st.video(os.path.join("temp", "output.mp4"))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                # Print the full traceback to the console for debugging
                print("Full error traceback:")
                print(traceback.format_exc())

# Add some information about the project
st.markdown("---")
st.markdown("""
### About Wav2Lip
Wav2Lip is an AI model that can generate lip-synced videos from an image and an audio file. 
The model was trained on thousands of videos to learn how to match lip movements with speech.

### How to use
1. Upload an image containing a face
2. Upload an audio file (WAV or MP3)
3. Click "Generate Lip Sync Video"
4. Wait for the processing to complete
5. Download the generated video

### Tips
- Use clear, front-facing images for best results
- Audio should be clear and well-recorded
- Processing time depends on the length of the audio
""") 