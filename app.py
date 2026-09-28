import cv2
import streamlit as st
import time

from tracker import PersonTracker
from analytics import AttendanceAnalytics


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Campus Computer Vision",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE
# =========================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "analytics" not in st.session_state:
    st.session_state.analytics = None


# =========================================================
# HEADER
# =========================================================

st.title("🏫 Smart Campus Computer Vision")

st.caption(
    "Live Classroom Monitoring System | "
    "YOLO + BoT-SORT + Real-Time Analytics"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Controls")


video_source = st.sidebar.selectbox(
    "Select Video Source",
    [
        "video7.mp4",
        "Webcam"
    ]
)


fps_limit = st.sidebar.slider(
    "Processing FPS",
    min_value=1,
    max_value=10,
    value=5,
    step=1
)


confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.20,
    max_value=0.90,
    value=0.40,
    step=0.05
)


# =========================================================
# ENTRY / EXIT
# =========================================================

st.sidebar.subheader("🚪 Entry / Exit")


line_position = st.sidebar.slider(
    "Line Position",
    min_value=0.20,
    max_value=0.90,
    value=0.65,
    step=0.05
)


entry_direction = st.sidebar.selectbox(
    "Entry Direction",
    [
        "top_to_bottom",
        "bottom_to_top"
    ]
)


st.sidebar.divider()


# =========================================================
# BUTTONS
# =========================================================

start_button = st.sidebar.button(
    "▶️ Start",
    use_container_width=True
)

stop_button = st.sidebar.button(
    "⏹️ Stop",
    use_container_width=True
)

reset_button = st.sidebar.button(
    "🔄 Reset Analytics",
    use_container_width=True
)


# =========================================================
# BUTTON ACTIONS
# =========================================================

if start_button:

    st.session_state.running = True

    st.session_state.analytics = AttendanceAnalytics(
        line_position=line_position,
        entry_direction=entry_direction
    )


if stop_button:

    st.session_state.running = False


if reset_button:

    if st.session_state.analytics is not None:

        st.session_state.analytics.reset()

    st.success(
        "Analytics reset successfully."
    )


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    attendance_box = st.empty()


with col2:

    unique_box = st.empty()


with col3:

    room_box = st.empty()


with col4:

    quality_box = st.empty()


# =========================================================
# MAIN LAYOUT
# =========================================================

st.divider()


video_col, status_col = st.columns(
    [3, 1]
)


# =========================================================
# VIDEO
# =========================================================

with video_col:

    st.subheader("📹 Live Stream")

    video_placeholder = st.empty()


# =========================================================
# STATUS
# =========================================================

with status_col:

    st.subheader("📊 Status")

    detection_box = st.empty()

    event_box = st.empty()

    tracking_box = st.empty()

    line_box = st.empty()


# =========================================================
# WAITING STATE
# =========================================================

if not st.session_state.running:

    attendance_box.metric(
        "👥 Currently Present",
        0
    )

    unique_box.metric(
        "🔢 Unique Entries",
        0
    )

    room_box.metric(
        "🏫 Room",
        "Waiting"
    )

    quality_box.metric(
        "📷 Frame Quality",
        "Waiting"
    )

    detection_box.info(
        "👤 Persons Detected: Waiting"
    )

    event_box.info(
        "ℹ️ Event: Waiting..."
    )

    tracking_box.info(
        "🎯 Tracking: Not started"
    )

    line_box.info(
        "🚪 Entry/Exit Line: Not active"
    )


# =========================================================
# LIVE PROCESSING
# =========================================================

if st.session_state.running:

    # -----------------------------------------------------
    # LOAD TRACKER
    # -----------------------------------------------------

    with st.spinner(
        "Loading YOLO + BoT-SORT..."
    ):

        tracker = PersonTracker(
            model_name="yolo11n.pt"
        )


    # -----------------------------------------------------
    # ANALYTICS
    # -----------------------------------------------------

    if st.session_state.analytics is None:

        st.session_state.analytics = AttendanceAnalytics(
            line_position=line_position,
            entry_direction=entry_direction
        )


    analytics = st.session_state.analytics


    # -----------------------------------------------------
    # VIDEO SOURCE
    # -----------------------------------------------------

    if video_source == "video7.mp4":

        cap = cv2.VideoCapture(
            "video7.mp4"
        )

    else:

        cap = cv2.VideoCapture(
            0
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )


    # -----------------------------------------------------
    # CHECK SOURCE
    # -----------------------------------------------------

    if not cap.isOpened():

        st.error(
            "❌ Unable to open video source."
        )

        st.session_state.running = False

        st.stop()


    frame_delay = 1 / fps_limit


    # =====================================================
    # FRAME LOOP
    # =====================================================

    while cap.isOpened():

        loop_start = time.time()


        # -------------------------------------------------
        # READ FRAME
        # -------------------------------------------------

        ret, frame = cap.read()


        if not ret:

            st.info(
                "🎬 Video stream ended."
            )

            break


        # -------------------------------------------------
        # YOLO + BOtSORT
        # -------------------------------------------------

        result = tracker.track(
            frame
        )


        # -------------------------------------------------
        # ANALYTICS
        # -------------------------------------------------

        (
            current_present,
            unique_entries,
            last_event,
            current_ids,
            line_y
        ) = analytics.update(
            frame,
            result.boxes
        )


        # -------------------------------------------------
        # PERSON COUNT
        # -------------------------------------------------

        if result.boxes is not None:

            person_count = len(
                result.boxes
            )

        else:

            person_count = 0


        # =================================================
        # ANNOTATED FRAME
        # =================================================

        annotated_frame = result.plot()


        # =================================================
        # ENTRY / EXIT LINE
        # =================================================

        cv2.line(
            annotated_frame,
            (0, line_y),
            (
                annotated_frame.shape[1],
                line_y
            ),
            (0, 255, 255),
            3
        )


        cv2.putText(
            annotated_frame,
            "ENTRY / EXIT LINE",
            (
                20,
                max(
                    30,
                    line_y - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )


        # =================================================
        # ROOM STATUS
        # =================================================

        if current_present > 0:

            room_status = "OCCUPIED"

        else:

            room_status = "EMPTY"


        # =================================================
        # FRAME QUALITY
        # =================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        sharpness = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()


        if sharpness > 100:

            frame_quality = "Clear"

        elif sharpness > 40:

            frame_quality = "Moderate"

        else:

            frame_quality = "Blurry"


        # =================================================
        # VIDEO OVERLAY
        # =================================================

        cv2.putText(
            annotated_frame,
            f"Room: {room_status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        cv2.putText(
            annotated_frame,
            f"Present: {current_present}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )


        cv2.putText(
            annotated_frame,
            f"Unique Entries: {unique_entries}",
            (20, 101),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0),
            2
        )


        # =================================================
        # CURRENT TRACKING IDS
        # =================================================

        sorted_ids = sorted(
            list(current_ids)
        )


        if sorted_ids:

            id_text = (
                "Tracking IDs: "
                + ", ".join(
                    map(
                        str,
                        sorted_ids
                    )
                )
            )

            cv2.putText(
                annotated_frame,
                id_text,
                (20, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 0),
                2
            )


        # =================================================
        # DISPLAY
        # =================================================

        display_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )


        video_placeholder.image(
            display_frame,
            channels="RGB",
            use_container_width=True
        )


        # =================================================
        # TOP METRICS
        # =================================================

        attendance_box.metric(
            "👥 Currently Present",
            current_present
        )


        unique_box.metric(
            "🔢 Unique Entries",
            unique_entries
        )


        room_box.metric(
            "🏫 Room",
            room_status
        )


        quality_box.metric(
            "📷 Frame Quality",
            frame_quality
        )


        # =================================================
        # STATUS
        # =================================================

        detection_box.info(
            f"👤 Persons Detected: "
            f"{person_count}"
        )


        # =================================================
        # EVENT
        # =================================================

        if "Entry detected" in last_event:

            event_box.success(
                f"🟢 {last_event}"
            )

        elif "Exit detected" in last_event:

            event_box.warning(
                f"🔴 {last_event}"
            )

        else:

            event_box.info(
                f"ℹ️ {last_event}"
            )


        # =================================================
        # TRACKING
        # =================================================

        if current_ids:

            tracking_box.success(
                "🎯 Tracking Active | IDs: "
                + ", ".join(
                    map(
                        str,
                        sorted_ids
                    )
                )
            )

        else:

            tracking_box.warning(
                "⚠️ No tracking IDs"
            )


        # =================================================
        # LINE
        # =================================================

        line_box.info(
            f"🚪 Line: "
            f"{int(line_position * 100)}%"
            f" | Entry: "
            f"{entry_direction}"
        )


        # =================================================
        # FPS LIMIT
        # =================================================

        elapsed = (
            time.time()
            - loop_start
        )

        time.sleep(
            max(
                0,
                frame_delay - elapsed
            )
        )


    # =====================================================
    # RELEASE
    # =====================================================

    cap.release()