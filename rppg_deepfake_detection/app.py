import tempfile
from pathlib import Path

import streamlit as st

from src.video_processor import process_video


st.set_page_config(page_title="rPPG Face Video Analyzer", page_icon="🎥", layout="wide")

st.title("rPPG Face Video Analyzer")
st.write(
    "Upload a video to detect faces and preview the face frames extracted for "
    "the rPPG pipeline."
)

video = st.file_uploader(
    "Choose a video",
    type=["mp4", "mov", "avi", "mkv"],
    help="For a quick preview, start with a short video where the face is clearly visible.",
)
max_frames = st.slider("Frames to analyze", min_value=30, max_value=600, value=150, step=30)

if video is not None:
    st.video(video)

    if st.button("Analyze video", type="primary"):
        suffix = Path(video.name).suffix or ".mp4"
        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_video:
                temporary_video.write(video.getbuffer())
                temporary_path = temporary_video.name

            with st.spinner("Scanning video frames for faces…"):
                face_frames = process_video(temporary_path, max_frames=max_frames)

            st.session_state["face_frames"] = face_frames
            st.session_state["video_name"] = video.name
        except Exception as error:
            st.error(f"The video could not be analyzed: {error}")
            st.session_state.pop("face_frames", None)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

if "face_frames" in st.session_state:
    face_frames = st.session_state["face_frames"]
    st.subheader(f"Results for {st.session_state['video_name']}")

    if not face_frames:
        st.warning("No face frames were found. Try a video with a larger, well-lit face.")
    else:
        st.success(f"Found {len(face_frames)} face frames in the analyzed portion of the video.")
        st.caption("These cropped frames can be passed to the rPPG signal-processing stage.")

        preview_frames = face_frames[:8]
        columns = st.columns(4)
        for index, frame in enumerate(preview_frames):
            with columns[index % len(columns)]:
                st.image(frame, channels="BGR", caption=f"Face frame {index + 1}", use_container_width=True)
