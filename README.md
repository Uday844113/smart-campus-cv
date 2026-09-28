# Smart Campus Computer Vision

A real-time classroom monitoring system using computer vision for
person detection, tracking, occupancy monitoring, frame-quality analysis,
and configurable entry/exit detection.

## Features

- Real-time person detection
- Multi-person tracking
- Currently Present count
- Room Occupied / Empty status
- Frame quality detection
- Configurable entry/exit line
- Unique entry event tracking
- Live Stream UI using Streamlit
- YOLO + BoT-SORT tracking
- Webcam and video-file input
- Frame-by-frame processing

## Technology Stack

- Python
- OpenCV
- YOLO
- Ultralytics
- BoT-SORT
- Streamlit
- NumPy

## Project Structure

smart-campus-cv/
│
├── app.py
├── tracker.py
├── analytics.py
├── utils.py
├── requirements.txt
├── README.md
├── video7.mp4
└── venv/

## Architecture

Camera / Video Stream
        |
        v
   Frame Capture
        |
        v
      YOLO
        |
        v
 Person Detection
        |
        v
   BoT-SORT
        |
        v
 Person Tracking
        |
        +------------------+
        |                  |
        v                  v
 Currently Present    Entry / Exit
        |                  |
        v                  v
 Room Occupancy      Unique Entries
        |
        v
   Streamlit UI

## Installation

Create and activate a virtual environment:

    python -m venv venv

Windows CMD:

    venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

## Run

Start the application:

    streamlit run app.py

The application opens a Streamlit dashboard.

## Input Sources

The application supports:

1. video7.mp4
2. Webcam

The processing is performed frame-by-frame rather than processing the
complete video as a batch.

## Currently Present

Currently Present represents the number of people actively tracked
in the current frame/short tracking window.

This is different from the number of unique people observed during
the complete session.

## Room Occupancy

The room is considered:

- OCCUPIED when at least one person is currently tracked.
- EMPTY when no person is currently tracked.

## Entry / Exit

A configurable line-crossing mechanism is implemented.

The line position and entry direction can be configured from the
Streamlit sidebar.

For reliable entry/exit validation, the camera view should contain
a visible entrance or exit boundary.

The supplied classroom video does not contain a dedicated doorway,
so entry/exit results from that video should not be interpreted as
validated doorway events.

## Frame Quality

Frame quality is estimated using image sharpness.

The application classifies frames approximately as:

- Clear
- Moderate
- Blurry

This is intended as a lightweight quality indicator rather than a
complete video-quality assessment system.

## Real-Time Processing

The application processes frames sequentially and supports a
configurable processing rate.

The processing FPS can be adjusted from the Streamlit sidebar.

## Limitations

- Tracking IDs are not guaranteed to remain identical after long
  disappearance or severe occlusion.
- Similar-looking people can make identity association difficult.
- Entry/exit detection requires a camera view containing a suitable
  entrance boundary.
- Heavy blur, occlusion, poor lighting, and crowded scenes can reduce
  detection and tracking accuracy.
- Currently Present represents tracked visibility, not a physical
  guarantee that every person is inside the room.

## Scaling Considerations

For a larger deployment, inference should be separated from the UI
and camera ingestion layer.

A scalable architecture would use:

Camera Streams
    |
    v
Stream Ingestion
    |
    v
GPU Inference Workers
    |
    v
Tracking / Analytics
    |
    v
Event Queue
    |
    v
Central Database
    |
    v
Monitoring Dashboard

Multiple inference workers can process camera streams in parallel.
Frame sampling can also reduce unnecessary computation.

## Development Test Video

The project includes:

    video7.mp4

This video is used for classroom detection, tracking, occupancy, and
frame-quality testing.

## Author

Computer Vision Engineer Hiring Task