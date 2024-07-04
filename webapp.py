import os
import cv2
import streamlit as st
import torch
import numpy as np
from ultralytics import YOLO
import tempfile
from pathlib import Path
import base64
from queue import Queue
from threading import Thread

# Check if CUDA is available and use GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    st.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    st.write("Using CPU")

# Initialize YOLO model
try:
    model = YOLO('yolov8x.pt').to(device)  # Load the model onto the selected device
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit interface
st.title("Object Detection Web Application with YOLOv8")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "mp4"])

def handle_image(file_buffer):
    try:
        file_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = model(img)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Processed Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")

def handle_video(file_buffer):
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_buffer.read())

        # Display the uploaded video
        st.video(tfile.name)

        cap = cv2.VideoCapture(tfile.name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            out = cv2.VideoWriter(temp_output.name, fourcc, 30.0, (frame_width, frame_height))

            def read_frames(cap, queue):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    queue.put(frame)
                cap.release()
                queue.put(None)

            def process_frames(queue_in, queue_out):
                while True:
                    frame = queue_in.get()
                    if frame is None:
                        queue_out.put(None)
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    results = model(frame_rgb)
                    res_plotted = results[0].plot()
                    queue_out.put(cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))  # Convert back to BGR for writing

            def write_frames(queue, out):
                frame_count = 0
                while True:
                    frame = queue.get()
                    if frame is None:
                        break
                    out.write(frame)
                    frame_count += 1
                out.release()
                return frame_count

            frame_queue = Queue(maxsize=10)
            processed_queue = Queue(maxsize=10)

            reader_thread = Thread(target=read_frames, args=(cap, frame_queue))
            processor_thread = Thread(target=process_frames, args=(frame_queue, processed_queue))
            writer_thread = Thread(target=write_frames, args=(processed_queue, out))

            reader_thread.start()
            processor_thread.start()
            writer_thread.start()

            reader_thread.join()
            processor_thread.join()
            frame_count = writer_thread.join()

        # Display processed video and frame count
        st.write(f"Processed {frame_count} frames")
        if os.path.exists(temp_output.name):
            st.video(temp_output.name)
            st.markdown(get_download_link(temp_output.name), unsafe_allow_html=True)
        else:
            st.error("Processed video file could not be found.")
    except Exception as e:
        st.error(f"Error processing video: {e}")

def handle_webcam():
    try:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_button_key = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame_rgb = np.array(frame_rgb)  # Ensure it's a NumPy array
            results = model(frame_rgb)
            res_plotted = results[0].plot()

            stframe.image(res_plotted, channels="RGB", use_column_width=True)

            # Stop the webcam feed if user clicks the stop button
            if st.button("Stop Webcam", key=f"stop_button_{stop_button_key}"):
                break
            stop_button_key += 1

        cap.release()
    except Exception as e:
        st.error(f"Error processing webcam feed: {e}")

def get_download_link(file_path):
    file_name = Path(file_path).name
    return f'<a href="data:application/octet-stream;base64,{get_base64(file_path)}" download="{file_name}">Download Processed Video</a>'

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Add an option to select webcam feed
if st.button("Start Webcam"):
    handle_webcam()

if uploaded_file is not None:
    try:
        file_extension = uploaded_file.name.rsplit('.', 1)[1].lower()
        if file_extension == 'jpg':
            handle_image(uploaded_file)
        elif file_extension == 'mp4':
            with st.spinner('Video preprocessing...'):
                handle_video(uploaded_file)
        else:
            st.error("Unsupported file type")
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
