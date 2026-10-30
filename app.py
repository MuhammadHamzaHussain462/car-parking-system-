from pathlib import Path
import tempfile
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import cv2

from src.detector import load_detector, detect_image, model_info
from src.parking import analyze_grid, draw_grid
from src.video_processor import process_video
from src.analytics import build_summary, class_count_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "weights.pt"

st.set_page_config(page_title="SmartPark AI", page_icon="🅿️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#0b1020; --panel:#121a2d; --line:#24324b; --muted:#91a0b8; --cyan:#38d6c4; --gold:#e6b566; }
.stApp { background: radial-gradient(circle at 80% 0%, #172743 0, #0b1020 42%); color:#edf3ff; font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#0d1526; border-right:1px solid #24324b; }
.block-container { padding-top:1.6rem; max-width:1500px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.4px; }
.hero { padding:26px 30px; border:1px solid #2a3b58; border-radius:22px; background:linear-gradient(115deg,rgba(28,52,79,.96),rgba(18,26,45,.94)); margin-bottom:22px; }
.eyebrow { color:#38d6c4; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk'; font-size:34px; font-weight:700; margin:5px 0; color:#f4f7ff; }
.hero-sub { color:#a9b8ce; font-size:14px; }
.metric { background:linear-gradient(145deg,#16243a,#111a2c); border:1px solid #2a3a55; border-radius:16px; padding:17px 19px; min-height:112px; }
.metric-label { color:#9eacc2; font-size:12px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { font-family:'Space Grotesk'; font-size:29px; font-weight:700; color:#f5f8ff; margin-top:7px; }
.metric-note { color:#38d6c4; font-size:11px; margin-top:2px; }
.section-card { background:#121a2d; border:1px solid #263650; border-radius:16px; padding:18px; }
.small-muted { color:#9eacc2; font-size:13px; }
div.stButton > button { border-radius:10px; border:1px solid #38d6c4; background:#153b43; color:#eafffc; font-weight:700; min-height:43px; }
div.stButton > button:hover { background:#1c5960; border-color:#6af0df; color:white; }
.stTabs [data-baseweb="tab"] { color:#a9b8ce; }
.stTabs [aria-selected="true"] { color:#38d6c4 !important; }
[data-testid="stFileUploader"] { border:1px dashed #405878; border-radius:14px; padding:10px; }
hr { border-color:#263650; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, note="LIVE ANALYSIS"):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

def get_model():
    try:
        return load_detector(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.info("Place your trained RF-DETR checkpoint at models/weights.pt, then restart the app.")
        return None

st.sidebar.markdown("## 🅿️ SMARTPARK")
st.sidebar.caption("COMPUTER VISION • PARKING INTELLIGENCE")
page = st.sidebar.radio("NAVIGATION", ["Overview", "Image Detection", "Video Detection", "Parking Monitor", "About & Setup"])
st.sidebar.divider()
if MODEL_PATH.exists():
    st.sidebar.success("Model file found")
else:
    st.sidebar.warning("Model weights missing")
st.sidebar.caption("Local inference • No hosted prediction API")

if page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">AI PARKING INTELLIGENCE PLATFORM</div><div class="hero-title">See your parking lot differently.</div><div class="hero-sub">Detect vehicles, inspect occupancy and turn visual data into actionable parking insights.</div></div>', unsafe_allow_html=True)
    model_status = "Ready" if MODEL_PATH.exists() else "Needs weights"
    a,b,c,d = st.columns(4)
    with a: metric_card("AI MODEL", model_status, "LOCAL MODEL")
    with b: metric_card("INPUT MODES", "02", "IMAGE + VIDEO")
    with c: metric_card("PARKING GRID", "Custom", "ROW / SLOT SETUP")
    with d: metric_card("INFERENCE", "On-device", "NO CLOUD API")
    st.write("")
    st.markdown("### Your workflow")
    cols = st.columns(3)
    cards = [
        ("01","Upload","Choose a parking image or recorded video."),
        ("02","Detect","Run your trained model and inspect bounding boxes."),
        ("03","Understand","Review vehicle counts and estimated grid occupancy.")
    ]
    for col,(num,title,desc) in zip(cols,cards):
        with col:
            st.markdown(f'<div class="section-card"><div class="eyebrow">{num} / WORKFLOW</div><h3>{title}</h3><div class="small-muted">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    if not MODEL_PATH.exists():
        st.warning("Place your trained RF-DETR checkpoint at models/weights.pt to enable local detection.")
    else:
        st.success("Model weights detected. Open Image Detection to run your first test.")

elif page == "Image Detection":
    st.markdown("## Image Detection")
    st.caption("Upload a parking-lot image and run local object detection.")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp"], key="image_upload")
    left,right = st.columns([1,1])
    with left:
        conf = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        imgsz = st.select_slider("Inference image size", options=[320,416,512,640,768], value=640)
        run = st.button("▶  RUN IMAGE DETECTION", use_container_width=True, disabled=uploaded is None)
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with right:
            st.image(image, caption="Original image", use_container_width=True)
        if run:
            model = get_model()
            if model is not None:
                with st.spinner("Analyzing image with your local AI model..."):
                    result_img, detections = detect_image(model, image, conf=conf, imgsz=imgsz)
                st.success(f"Analysis complete — {len(detections)} object(s) detected.")
                x,y,z = st.columns(3)
                with x: metric_card("DETECTIONS", len(detections))
                with y: metric_card("CLASSES", detections["class_name"].nunique() if not detections.empty else 0)
                with z: metric_card("AVG CONFIDENCE", f'{detections["confidence"].mean():.1%}' if not detections.empty else "—")
                st.markdown("### Annotated result")
                st.image(result_img, use_container_width=True)
                if not detections.empty:
                    st.markdown("### Detection details")
                    st.dataframe(detections, use_container_width=True, hide_index=True)
                    st.download_button("⬇ Download detection CSV", detections.to_csv(index=False).encode("utf-8"), "smartpark_image_detections.csv", "text/csv")
                else:
                    st.info("No objects passed the selected confidence threshold.")

elif page == "Video Detection":
    st.markdown("## Video Detection")
    st.caption("Process a recorded parking video locally. Processing every frame may take time on CPU.")
    video = st.file_uploader("Choose a video", type=["mp4","mov","avi","mkv"], key="video_upload")
    c1,c2,c3 = st.columns(3)
    with c1: conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05, key="vconf")
    with c2: stride = st.slider("Analyze every Nth frame", 1, 10, 2)
    with c3: max_width = st.selectbox("Processing width", [480,640,800,960], index=1)
    if video and st.button("▶  PROCESS VIDEO", use_container_width=True):
        model = get_model()
        if model is not None:
            suffix = Path(video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video.getbuffer())
                input_path = tmp.name
            output_path = str(ROOT / "assets" / "outputs" / "smartpark_annotated_video.mp4")
            progress = st.progress(0, text="Preparing video...")
            try:
                summary = process_video(model, input_path, output_path, conf=conf, stride=stride, max_width=max_width, progress=progress)
                st.success("Video processing completed.")
                st.video(output_path)
                p,q,r = st.columns(3)
                with p: metric_card("FRAMES PROCESSED", summary["frames_processed"])
                with q: metric_card("TOTAL DETECTIONS", summary["total_detections"])
                with r: metric_card("VIDEO FPS", f'{summary["fps"]:.1f}')
                if summary["class_counts"]:
                    st.dataframe(pd.DataFrame([{"Vehicle / Object":k,"Detections":v} for k,v in summary["class_counts"].items()]), use_container_width=True, hide_index=True)
                with open(output_path,"rb") as f:
                    st.download_button("⬇ Download annotated video", f.read(), "smartpark_annotated_video.mp4", "video/mp4")
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                try: os.remove(input_path)
                except Exception: pass

elif page == "Parking Monitor":
    st.markdown("## Parking Space Monitor")
    st.caption("Grid-based occupancy estimate. Set the grid to match the visible parking bays in your camera image.")
    image_file = st.file_uploader("Upload a top/front view parking image", type=["jpg","jpeg","png"], key="parking_image")
    a,b,c = st.columns(3)
    with a: rows = st.number_input("Parking rows", min_value=1, max_value=12, value=3)
    with b: cols = st.number_input("Slots per row", min_value=1, max_value=20, value=6)
    with c: conf = st.slider("Detection confidence", 0.05, 0.95, 0.35, 0.05, key="pconf")
    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Uploaded parking image", use_container_width=True)
        if st.button("▶  ANALYZE PARKING OCCUPANCY", use_container_width=True):
            model = get_model()
            if model is not None:
                with st.spinner("Detecting vehicles and estimating grid occupancy..."):
                    _, det = detect_image(model, img, conf=conf, imgsz=640)
                    occupancy = analyze_grid(det, img.width, img.height, int(rows), int(cols))
                    marked = draw_grid(img, occupancy, int(rows), int(cols))
                st.image(marked, caption="Green = estimated empty • Red = estimated occupied", use_container_width=True)
                a,b,c = st.columns(3)
                with a: metric_card("TOTAL SLOTS", occupancy["total"])
                with b: metric_card("OCCUPIED", occupancy["occupied"])
                with c: metric_card("AVAILABLE", occupancy["available"])
                st.metric("Estimated occupancy", f'{occupancy["occupancy_rate"]:.1f}%')
                st.dataframe(pd.DataFrame(occupancy["slot_rows"]), use_container_width=True, hide_index=True)
                st.warning("This is a configurable rectangular-grid estimate, not calibrated stall-level tracking. For accurate real parking deployment, define each stall polygon against a fixed camera view.")
else:
    st.markdown("## About SmartPark AI")
    st.markdown("""
    **SmartPark AI** is a computer-vision assignment prototype for vehicle detection and parking-lot analysis.

    **Architecture**
    - Roboflow: dataset preparation and RF-DETR model training.
    - Local RF-DETR Small inference: loads `models/weights.pt` on your computer.
    - OpenCV: image/video decoding and annotated output.
    - Streamlit: interactive dashboard.
    - Pandas + Plotly: tabular results and analytics.

    **Model setup**
    1. Copy your trained RF-DETR Small checkpoint into the `models/` folder.
    2. Save the checkpoint as `models/weights.pt`.
    3. Install packages from `requirements.txt`.
    4. Run `streamlit run app.py`.

    The app performs local inference and does not call a hosted Roboflow prediction API.
    """)
    info = model_info(str(MODEL_PATH))
    st.json(info)
    st.caption("Parking occupancy currently uses a rectangular grid approximation. Explain this limitation in your report.")
