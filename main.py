import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load YOLOv5 model (pretrained)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Video processing function
def process_frame(frame, model):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image for YOLOv5 input
    pil_img = Image.fromarray(rgb_frame)

    # Perform detection
    results = model(pil_img)

    # Get bounding boxes and labels
    boxes = results.xyxy[0].cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Draw rectangle and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

# Streamlit UI
st.title("Real-Time Traffic Object Detection using YOLOv5 and GPU")

# File uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Video capture
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()  # Placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with YOLOv5 model
        frame = process_frame(frame, model)

        # Convert frame to image format for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels='RGB')

    cap.release()
