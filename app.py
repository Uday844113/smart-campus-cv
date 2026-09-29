import time
from pathlib import Path

import cv2
import streamlit as st

from tracker import PersonTracker
from analytics import AttendanceAnalytics


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Campus Computer Vision",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0;
        }

        .subtitle {
            color: #7b8190;
            font-size: 16px;
            margin-top: 5px;
        }

        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 18px;
            border: 1px solid #e5e7eb;
            text-align: center;
        }

        .metric-title {
            color: #6b7280;
            font-size: 14px;
        }

        .metric-value {
            color: #111827;
            font-size: 30px;
            font-weight: 700;
        }

        .status-box {
            padding: 12px 16px;
            border-radius: 10px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "video7.mp4"


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "analytics" not in st.session_state:
    st.session_state.analytics = None

if "tracker" not in st.session_state:
    st.session_state.tracker = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏫 Smart Campus Computer Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Live Classroom Monitoring System | YOLO + BoT-SORT + Real-Time Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.header("⚙️ Controls")

    source = st.selectbox(
        "Select Video Source",
        ["video7.mp4", "Webcam"],
        index=0
    )

    processing_fps = st.slider(
        "Processing FPS",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05
    )

    st.markdown("### 🚪 Entry / Exit")

    line_position = st.slider(
        "Line Position",
        min_value=0.20,
        max_value=0.90,
        value=0.65,
        step=0.05
    )

    entry_direction = st.selectbox(
        "Entry Direction",
        ["top_to_bottom", "bottom_to_top"],
        index=0
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        start_button = st.button(
            "▶ Start",
            use_container_width=True
        )

    with col2:
        stop_button = st.button(
            "⏹ Stop",
            use_container_width=True
        )

    reset_button = st.button(
        "🔄 Reset Analytics",
        use_container_width=True
    )

    st.divider()

    st.caption("Cloud Demo")
    st.info(
        "For Streamlit Cloud, use video7.mp4. "
        "The Webcam option is intended for local execution."
    )


# ============================================================
# BUTTON ACTIONS
# ============================================================

if start_button:
    st.session_state.running = True

if stop_button:
    st.session_state.running = False

if reset_button:

    if st.session_state.analytics is not None:
        st.session_state.analytics.reset()

    st.session_state.running = False

    st.success("Analytics reset successfully.")


# ============================================================
# CREATE TRACKER / ANALYTICS
# ============================================================

if st.session_state.tracker is None:

    try:
        st.session_state.tracker = PersonTracker(
            model_name="yolo11n.pt"
        )
    except Exception as e:
        st.error(f"Unable to load YOLO model: {e}")
        st.stop()


if st.session_state.analytics is None:

    st.session_state.analytics = AttendanceAnalytics(
        line_position=line_position,
        entry_direction=entry_direction
    )


# Update analytics configuration

st.session_state.analytics.line_position = line_position
st.session_state.analytics.entry_direction = entry_direction


tracker = st.session_state.tracker
analytics = st.session_state.analytics


# ============================================================
# TOP METRICS
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    present_placeholder = st.empty()

with metric2:
    entries_placeholder = st.empty()

with metric3:
    room_placeholder = st.empty()

with metric4:
    quality_placeholder = st.empty()


# ============================================================
# LIVE STREAM / STATUS
# ============================================================

stream_col, status_col = st.columns([1.8, 1])

with stream_col:

    st.subheader("📹 Live Stream")

    video_placeholder = st.empty()

with status_col:

    st.subheader("📊 Status")

    persons_placeholder = st.empty()

    event_placeholder = st.empty()

    tracking_placeholder = st.empty()

    line_placeholder = st.empty()


# ============================================================
# INITIAL DISPLAY
# ============================================================

if not st.session_state.running:

    present_placeholder.metric(
        "Currently Present",
        0
    )

    entries_placeholder.metric(
        "Unique Entries",
        0
    )

    room_placeholder.metric(
        "Room",
        "Waiting"
    )

    quality_placeholder.metric(
        "Frame Quality",
        "Waiting"
    )

    persons_placeholder.info(
        "Press ▶ Start to begin processing."
    )

    event_placeholder.info(
        "Event: Waiting..."
    )

    tracking_placeholder.info(
        "Tracking IDs: --"
    )

    line_placeholder.info(
        f"Entry/Exit Line: {line_position:.2f}"
    )


# ============================================================
# VIDEO SOURCE
# ============================================================

def open_video_source(selected_source):

    if selected_source == "video7.mp4":

        if not VIDEO_PATH.exists():

            st.error(
                f"Video file not found: {VIDEO_PATH}"
            )

            st.stop()

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

    else:

        cap = cv2.VideoCapture(0)

        # Webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


# ============================================================
# FRAME QUALITY
# ============================================================

def get_frame_quality(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if sharpness > 100:
        return "Clear"

    elif sharpness > 40:
        return "Moderate"

    else:
        return "Blurry"


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_overlay(
    frame,
    present,
    unique_entries,
    room_status,
    quality,
    tracking_ids,
    line_y
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Entry / Exit line
    # --------------------------------------------------------

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "ENTRY / EXIT LINE",
        (20, max(line_y - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Background panel
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (15, 15),
        (360, 155),
        (20, 20, 20),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Present: {present}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique Entries: {unique_entries}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Room: {room_status}",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Quality: {quality}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Tracking IDs
    # --------------------------------------------------------

    if tracking_ids:

        ids_text = ", ".join(
            str(track_id)
            for track_id in sorted(tracking_ids)
        )

        if len(ids_text) > 45:
            ids_text = ids_text[:45] + "..."

        cv2.putText(
            frame,
            f"IDs: {ids_text}",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return frame


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

if st.session_state.running:

    cap = open_video_source(source)

    if not cap.isOpened():

        source_name = (
            str(VIDEO_PATH)
            if source == "video7.mp4"
            else "Webcam"
        )

        st.error(
            f"Unable to open video source: {source_name}"
        )

        st.session_state.running = False

        st.stop()

    # --------------------------------------------------------
    # Video FPS
    # --------------------------------------------------------

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:
        source_fps = 30

    # Delay for requested processing FPS
    frame_delay = 1.0 / processing_fps

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    while st.session_state.running:

        loop_start = time.time()

        ret, frame = cap.read()

        # ----------------------------------------------------
        # Video ended
        # ----------------------------------------------------

        if not ret:

            if source == "video7.mp4":

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0
                )

                continue

            else:

                st.warning(
                    "Unable to read frame from webcam."
                )

                break

        # ----------------------------------------------------
        # YOLO + BoT-SORT
        # ----------------------------------------------------

        try:

            result = tracker.model.track(
                frame,
                persist=True,
                classes=[0],
                conf=confidence,
                tracker="botsort_reid.yaml",
                verbose=False
            )[0]

        except Exception as e:

            st.error(
                f"Tracking error: {e}"
            )

            break

        # ----------------------------------------------------
        # Analytics
        # ----------------------------------------------------

        try:

            present, unique_entries, event, tracking_ids, line_y = (
                analytics.update(
                    frame,
                    result.boxes
                )
            )

        except Exception as e:

            st.error(
                f"Analytics error: {e}"
            )

            break

        # ----------------------------------------------------
        # Frame Quality
        # ----------------------------------------------------

        quality = get_frame_quality(frame)

        # ----------------------------------------------------
        # Room Status
        # ----------------------------------------------------

        if present > 0:
            room_status = "Occupied"
        else:
            room_status = "Empty"

        # ----------------------------------------------------
        # Draw YOLO boxes
        # ----------------------------------------------------

        annotated_frame = result.plot()

        # ----------------------------------------------------
        # Draw application overlay
        # ----------------------------------------------------

        annotated_frame = draw_overlay(
            annotated_frame,
            present,
            unique_entries,
            room_status,
            quality,
            tracking_ids,
            line_y
        )

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Display video
        # ----------------------------------------------------

        video_placeholder.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

        # ----------------------------------------------------
        # Update metrics
        # ----------------------------------------------------

        present_placeholder.metric(
            "Currently Present",
            present
        )

        entries_placeholder.metric(
            "Unique Entries",
            unique_entries
        )

        room_placeholder.metric(
            "Room",
            room_status
        )

        quality_placeholder.metric(
            "Frame Quality",
            quality
        )

        # ----------------------------------------------------
        # Update status
        # ----------------------------------------------------

        persons_placeholder.markdown(
            f"""
            <div class="status-box">
            <b>Persons Detected:</b> {present}
            </div>
            """,
            unsafe_allow_html=True
        )

        event_placeholder.markdown(
            f"""
            <div class="status-box">
            <b>Event:</b> {event}
            </div>
            """,
            unsafe_allow_html=True
        )

        if tracking_ids:

            ids_text = ", ".join(
                str(x)
                for x in sorted(tracking_ids)
            )

            tracking_placeholder.markdown(
                f"""
                <div class="status-box">
                <b>Tracking IDs:</b><br>
                {ids_text}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            tracking_placeholder.markdown(
                """
                <div class="status-box">
                <b>Tracking IDs:</b> None
                </div>
                """,
                unsafe_allow_html=True
            )

        line_placeholder.markdown(
            f"""
            <div class="status-box">
            <b>Entry/Exit Line:</b>
            {line_position:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Maintain processing FPS
        # ----------------------------------------------------

        elapsed = time.time() - loop_start

        remaining = frame_delay - elapsed

        if remaining > 0:
            time.sleep(remaining)

    # --------------------------------------------------------
    # Release source
    # --------------------------------------------------------

    cap.release()