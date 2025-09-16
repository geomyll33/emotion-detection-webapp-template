import streamlit as st
from deepface import DeepFace
import cv2
import tempfile
import pandas as pd

st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="centered")
st.title("🎭 Emotion Detection from Video")
st.write("Ανεβάστε ένα video και ανιχνεύστε τα κυρίαρχα συναισθήματα καρέ‑καρέ.")

uploaded_file = st.file_uploader("📤 Επιλέξτε ένα video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)  # preview του video

    if st.button("🔎 Εκκίνηση Ανάλυσης"):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        emotions = []

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotions.append(result[0]['dominant_emotion'])
                except Exception:
                    pass

            progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()

        st.success("✅ Ανάλυση Ολοκληρώθηκε!")
        df = pd.Series(emotions).value_counts()
        st.bar_chart(df)
