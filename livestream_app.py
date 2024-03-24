import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import os
import asyncio
from ultralytics import YOLO

async def main():
    # Streamlit web app
    st.title("Object Tracking")

    # Radio button for user selection
    option = st.radio("Choose an option:", ("Live Stream", "Upload Video"))

    if option == "Live Stream":
        # Initialize webcam
        cap = cv2.VideoCapture(0)

        # Button to start the stream
        start_button_pressed = st.button("Start Live Stream")

        # Placeholder for video frame
        frame_placeholder = st.empty()

        # Button to stop the stream
        stop_button_pressed = st.button("Stop")

        # Check if the start button is pressed
        if start_button_pressed:
            # Call the function to capture video with the stop button state
            await capture_video(cap, stop_button_pressed, frame_placeholder)

    elif option == "Upload Video":
        # File uploader for video upload
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        # Button to start tracking
        start_button_pressed = st.button("Start Tracking")

        # Placeholder for video frame
        frame_placeholder = st.empty()

        # Button to stop tracking
        stop_button_pressed = st.button("Stop")

        # Check if the start button is pressed and file is uploaded
        if start_button_pressed and uploaded_file is not None:
            # Call the function to track uploaded video with the stop button state
            await track_uploaded_video(uploaded_file, stop_button_pressed, frame_placeholder)

        # Release resources
        if uploaded_file:
            uploaded_file.close()

# Function to capture video stream
async def capture_video(cap, stop_button, frame_placeholder):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    frame_count = 0
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended.")
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            # Resize frame to reduce processing time
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Detect and track objects using YOLOv8
            results = model.track(frame_rgb, persist=True)

            # Plot results
            frame_ = results[0].plot()

            # Display frame with bounding boxes
            frame_placeholder.image(frame_, channels="RGB")

        frame_count += 1
        await asyncio.sleep(0)  # Allow other tasks to run

    # Release resources
    cap.release()

# Function to perform object tracking on uploaded video
async def track_uploaded_video(video_file, stop_button, frame_placeholder):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Create a temporary file to save the uploaded video
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()

    # OpenCV's VideoCapture for reading video file
    cap = cv2.VideoCapture(temp_video.name)

    frame_count = 0
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        
        if not ret:
            st.write("The video capture has ended.")
            break

        # Process every 5th frame
        if frame_count % 5 == 0:
            # Resize frame to reduce processing time
            frame_resized = cv2.resize(frame, (640, 480))

            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Detect and track objects using YOLOv8
            results = model.track(frame_rgb, persist=True)

            # Plot results
            frame_ = results[0].plot()

            # Display frame with bounding boxes
            frame_placeholder.image(frame_, channels="RGB")

        frame_count += 1
        await asyncio.sleep(0)  # Allow other tasks to run

    # Release resources
    cap.release()
    # Remove temporary file
    os.remove(temp_video.name)

# Run the app
asyncio.run(main())
