import time
from pathlib import Path

import cv2
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

from tracker import PersonTracker
from analytics import AttendanceAnalytics


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Campus Computer Vision",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "video7.mp4"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-top: 4px;
    }

    .status-box {
        padding: 12px 15px;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #f8fafc;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "tracker" not in st.session_state:
    st.session_state.tracker = None

if "analytics" not in st.session_state:
    st.session_state.analytics = None

if "video_running" not in st.session_state:
    st.session_state.video_running = False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏫 Smart Campus Computer Vision</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Live Classroom Monitoring System | YOLO + BoT-SORT + Real-Time Analytics'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# LOAD TRACKER
# ============================================================

if st.session_state.tracker is None:

    try:
        st.session_state.tracker = PersonTracker(
            model_name="yolo11n.pt"
        )

    except Exception as e:
        st.error(f"Unable to load YOLO model: {e}")
        st.stop()


tracker = st.session_state.tracker


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Controls")

    source = st.selectbox(
        "Select Video Source",
        [
            "video7.mp4",
            "Webcam",
        ],
        index=0,
    )

    processing_fps = st.slider(
        "Processing FPS",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
    )

    st.markdown("### 🚪 Entry / Exit")

    line_position = st.slider(
        "Line Position",
        min_value=0.20,
        max_value=0.90,
        value=0.65,
        step=0.05,
    )

    entry_direction = st.selectbox(
        "Entry Direction",
        [
            "top_to_bottom",
            "bottom_to_top",
        ],
        index=0,
    )

    st.divider()

    start_button = st.button(
        "▶ Start",
        use_container_width=True,
    )

    stop_button = st.button(
        "⏹ Stop",
        use_container_width=True,
    )

    reset_button = st.button(
        "🔄 Reset Analytics",
        use_container_width=True,
    )

    st.divider()

    st.caption("Cloud Demo")

    st.info(
        "Video processing uses the selected FPS. "
        "Webcam inference is limited to approximately 5 FPS "
        "to reduce cloud CPU usage."
    )


# ============================================================
# ANALYTICS
# ============================================================

if st.session_state.analytics is None:

    st.session_state.analytics = AttendanceAnalytics(
        line_position=line_position,
        entry_direction=entry_direction,
    )


analytics = st.session_state.analytics

analytics.line_position = line_position
analytics.entry_direction = entry_direction


# ============================================================
# BUTTONS
# ============================================================

if start_button:
    st.session_state.video_running = True

if stop_button:
    st.session_state.video_running = False

if reset_button:

    analytics.reset()
    st.session_state.video_running = False

    st.success("Analytics reset successfully.")


# ============================================================
# FRAME QUALITY
# ============================================================

def get_frame_quality(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    if sharpness > 100:
        return "Clear"

    elif sharpness > 40:
        return "Moderate"

    return "Blurry"


# ============================================================
# SEATED / STANDING ESTIMATION
# ============================================================

def estimate_posture(boxes):

    seated = 0
    standing_moving = 0

    if boxes is None:
        return seated, standing_moving

    if boxes.xyxy is None:
        return seated, standing_moving

    xyxy = boxes.xyxy.cpu().tolist()

    for box in xyxy:

        x1, y1, x2, y2 = box

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        aspect_ratio = height / width

        # Prototype heuristic:
        # shorter/wider person box -> estimated seated
        # taller person box -> estimated standing/moving

        if aspect_ratio < 1.35:
            seated += 1
        else:
            standing_moving += 1

    return seated, standing_moving


# ============================================================
# DRAW OVERLAY
# ============================================================

def draw_overlay(
    frame,
    present,
    unique_entries,
    seated,
    standing_moving,
    room_status,
    quality,
    tracking_ids,
    line_y,
    event,
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # ENTRY / EXIT LINE
    # --------------------------------------------------------

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3,
    )

    cv2.putText(
        frame,
        "ENTRY / EXIT LINE",
        (20, max(30, line_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # INFO PANEL
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (500, 270),
        (20, 20, 20),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
    )

    cv2.putText(
        frame,
        f"Present: {present}",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Seated: {seated}",
        (25, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Standing/Moving: {standing_moving}",
        (25, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Unique Entries: {unique_entries}",
        (25, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Room: {room_status}",
        (25, 166),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Quality: {quality}",
        (25, 197),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    event_text = str(event)

    if len(event_text) > 48:
        event_text = event_text[:48] + "..."

    cv2.putText(
        frame,
        f"Event: {event_text}",
        (25, 228),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # TRACKING IDs
    # --------------------------------------------------------

    if tracking_ids:

        ids_text = ", ".join(
            str(track_id)
            for track_id in sorted(tracking_ids)
        )

        if len(ids_text) > 60:
            ids_text = ids_text[:60] + "..."

        cv2.putText(
            frame,
            f"IDs: {ids_text}",
            (25, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )

    return frame


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(frame):

    result = tracker.model.track(
        frame,
        persist=True,
        classes=[0],
        conf=confidence,
        tracker="botsort_reid.yaml",
        verbose=False,
    )[0]

    (
        present,
        unique_entries,
        event,
        tracking_ids,
        line_y,
    ) = analytics.update(
        frame,
        result.boxes,
    )

    seated, standing_moving = estimate_posture(
        result.boxes
    )

    quality = get_frame_quality(frame)

    room_status = (
        "Occupied"
        if present > 0
        else "Empty"
    )

    output = result.plot()

    output = draw_overlay(
        output,
        present,
        unique_entries,
        seated,
        standing_moving,
        room_status,
        quality,
        tracking_ids,
        line_y,
        event,
    )

    return (
        output,
        present,
        unique_entries,
        seated,
        standing_moving,
        event,
        tracking_ids,
        quality,
        room_status,
    )


# ============================================================
# VIDEO FILE
# ============================================================

def run_video_file():

    st.subheader("📹 Live Stream")

    video_placeholder = st.empty()

    st.subheader("📊 Status")

    c1, c2, c3, c4 = st.columns(4)

    present_placeholder = c1.empty()
    entries_placeholder = c2.empty()
    room_placeholder = c3.empty()
    quality_placeholder = c4.empty()

    c5, c6, c7 = st.columns(3)

    seated_placeholder = c5.empty()
    standing_placeholder = c6.empty()
    event_placeholder = c7.empty()

    ids_placeholder = st.empty()

    if not VIDEO_PATH.exists():

        st.error(
            f"video7.mp4 not found:\n{VIDEO_PATH}"
        )

        return

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        st.error(
            f"Unable to open video source:\n{VIDEO_PATH}"
        )

        return

    frame_delay = 1.0 / processing_fps

    while st.session_state.video_running:

        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            continue

        try:

            (
                output,
                present,
                unique_entries,
                seated,
                standing_moving,
                event,
                tracking_ids,
                quality,
                room_status,
            ) = process_frame(frame)

        except Exception as e:

            st.error(
                f"Processing error: {e}"
            )

            break

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB,
        )

        video_placeholder.image(
            output_rgb,
            channels="RGB",
            use_container_width=True,
        )

        present_placeholder.metric(
            "Currently Present",
            present,
        )

        entries_placeholder.metric(
            "Unique Entries",
            unique_entries,
        )

        room_placeholder.metric(
            "Room",
            room_status,
        )

        quality_placeholder.metric(
            "Frame Quality",
            quality,
        )

        seated_placeholder.metric(
            "Seated",
            seated,
        )

        standing_placeholder.metric(
            "Standing / Moving",
            standing_moving,
        )

        event_placeholder.metric(
            "Event",
            event,
        )

        if tracking_ids:

            ids_text = ", ".join(
                str(track_id)
                for track_id in sorted(
                    tracking_ids
                )
            )

        else:

            ids_text = "None"

        ids_placeholder.markdown(
            f"""
            <div class="status-box">
                <b>Tracking IDs:</b> {ids_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        elapsed = time.time() - loop_start

        sleep_time = frame_delay - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


# ============================================================
# WEBRTC WEBCAM PROCESSOR
# ============================================================

class SmartCampusVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.last_process_time = 0.0
        self.last_output = None

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        current_time = time.time()

        # ----------------------------------------------------
        # Limit YOLO inference to approximately 5 FPS.
        # This prevents processing every browser frame.
        # ----------------------------------------------------

        if (
            current_time - self.last_process_time
            < 0.20
        ):

            if self.last_output is not None:

                return frame.from_ndarray(
                    self.last_output,
                    format="bgr24",
                )

            return frame.from_ndarray(
                img,
                format="bgr24",
            )

        self.last_process_time = current_time

        try:

            (
                output,
                present,
                unique_entries,
                seated,
                standing_moving,
                event,
                tracking_ids,
                quality,
                room_status,
            ) = process_frame(img)

            self.last_output = output

            return frame.from_ndarray(
                output,
                format="bgr24",
            )

        except Exception as e:

            error_frame = img.copy()

            cv2.putText(
                error_frame,
                "Processing error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            self.last_output = error_frame

            return frame.from_ndarray(
                error_frame,
                format="bgr24",
            )


# ============================================================
# MAIN
# ============================================================

if source == "video7.mp4":

    if st.session_state.video_running:

        run_video_file()

    else:

        st.subheader("📹 Live Stream")

        st.info(
            "Select video7.mp4 and press ▶ Start."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Currently Present",
            0,
        )

        c2.metric(
            "Unique Entries",
            0,
        )

        c3.metric(
            "Room",
            "Waiting",
        )

        c4.metric(
            "Frame Quality",
            "Waiting",
        )

        c5, c6 = st.columns(2)

        c5.metric(
            "Seated",
            0,
        )

        c6.metric(
            "Standing / Moving",
            0,
        )


else:

    # ========================================================
    # WEBCAM
    # ========================================================

    st.subheader("📹 Live Webcam")

    st.info(
        "Click START and allow camera access. "
        "Move across the yellow Entry / Exit line "
        "to test entry and exit detection."
    )

    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    webrtc_streamer(
        key="smart-campus-webcam",
        video_processor_factory=(
            SmartCampusVideoProcessor
        ),
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    st.subheader(
        "📊 Webcam Analytics"
    )

    st.markdown(
        """
        <div class="status-box">

        <b>Entry / Exit Test</b><br><br>

        1. Stand above the yellow line.<br>
        2. Move completely across the line.<br>
        3. The video overlay will show
        <b>Entry detected — ID XX</b>.<br>
        4. Move back across the line to test exit.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="status-box">

        <b>Posture estimation:</b>
        Seated vs. Standing/Moving is a lightweight
        bounding-box based prototype estimate.
        A production system would use pose estimation
        for stronger posture accuracy.

        </div>
        """,
        unsafe_allow_html=True,
    )