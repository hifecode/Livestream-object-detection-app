import cv2
import streamlit as st
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Define a video transformer for object tracking
class ObjectTrackingTransformer(VideoTransformerBase):
    def __init__(self):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')

    async def recv(self, frame):
        # Convert frame to OpenCV format (BGR)
        frame_bgr = frame.to_ndarray(format="bgr24")

        # Resize frame to reduce processing time
        frame_resized = cv2.resize(frame_bgr, (640, 480))

        # Detect and track objects using YOLOv8
        results = self.model.track(frame_resized, persist=True)

        # Plot results
        frame_annotated = results[0].plot()

        # Convert frame back to RGB format
        frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)

        return av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")

# Streamlit web app
def main():
    # Set page title
    st.set_page_config(page_title="Object Tracking with Streamlit")

    # Streamlit web app
    st.title("Object Tracking")

    # Radio button for user selection
    option = st.radio("Choose an option:", ("Live Stream", "Upload Video"))

    if option == "Live Stream":
        # Start the WebRTC stream with object tracking
        # WebRTC streamer configuration
        rtc_configuration = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
        # Start the WebRTC stream with object tracking
        webrtc_streamer(key="live-stream", video_processor_factory=ObjectTrackingTransformer,
                        rtc_configuration=rtc_configuration)

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
            track_uploaded_video(uploaded_file, stop_button_pressed, frame_placeholder)

        # Release resources
        if uploaded_file:
            uploaded_file.close()

# Function to perform object tracking on uploaded video
def track_uploaded_video(video_file, stop_button, frame_placeholder):
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

            # Detect and track objects using YOLOv8
            results = model.track(frame_resized, persist=True)

            # Plot results
            frame_ = results[0].plot()

            # Display frame with bounding boxes
            frame_placeholder.image(frame_, channels="BGR")

        frame_count += 1

    # Release resources
    cap.release()
    # Remove temporary file
    os.remove(temp_video.name)

# Run the app
if __name__ == "__main__":
    main()
