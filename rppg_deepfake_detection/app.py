import tempfile
from pathlib import Path

import streamlit as st

from src.video_processor import analyze_video


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
                analysis = analyze_video(temporary_path, max_frames=max_frames)

            st.session_state["face_frames"] = analysis["face_frames"]
            st.session_state["quality_checks"] = analysis["checks"]
            st.session_state["overall_quality_status"] = analysis["overall_status"]
            st.session_state["frames_analyzed"] = analysis["frames_analyzed"]
            st.session_state["video_name"] = video.name
        except Exception as error:
            st.error(f"The video could not be analyzed: {error}")
            st.session_state.pop("face_frames", None)
            st.session_state.pop("quality_checks", None)
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

if "face_frames" in st.session_state:
    face_frames = st.session_state["face_frames"]
    st.subheader(f"Results for {st.session_state['video_name']}")

    st.subheader("Video quality checks")
    st.caption(f"Checks are based on up to {st.session_state['frames_analyzed']} sampled frames. These are screening indicators, not guarantees of rPPG accuracy.")
    if st.session_state.get("overall_quality_status") == "Ready for analysis":
        st.success("Ready for analysis")
    else:
        st.warning("May be unreliable — review the checks and suggestions below.")
    checks = st.session_state.get("quality_checks", {})
    columns = st.columns(len(checks)) if checks else []
    for column, (name, result) in zip(columns, checks.items()):
        with column:
            st.metric(name, result["status"])
            st.caption(result["value"])
    tips = [result["tip"] for result in checks.values() if result.get("tip")]
    if tips:
        st.markdown("**Suggestions**")
        for tip in tips:
            st.write(f"- {tip}")

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
