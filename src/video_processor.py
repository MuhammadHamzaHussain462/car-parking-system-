from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.detector import detections_to_frame


def process_video(
    model,
    input_path: str,
    output_path: str,
    conf=0.35,
    stride=2,
    max_width=640,
    progress=None,
):
    """Run local RF-DETR inference on sampled video frames and write an MP4."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("The uploaded video has invalid frame dimensions.")

    scale = min(1.0, float(max_width) / max(width, 1))
    out_w = max(2, int(width * scale))
    out_h = max(2, int(height * scale))
    out_w -= out_w % 2
    out_h -= out_h % 2

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create MP4 output. Check output folder and codec support.")

    frame_index = 0
    processed = 0
    total_detections = 0
    class_counts = Counter()
    last_annotated = None
    frame_stride = max(1, int(stride))

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if scale != 1.0:
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h))

            if frame_index % frame_stride == 0 or last_annotated is None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                detections = model.predict(
                    images=pil_image,
                    threshold=float(conf),
                    shape=(640, 640),
                )

                frame_detections = detections_to_frame(detections)
                annotated_rgb = pil_image.copy()
                from src.detector import annotate_image
                annotated_rgb = annotate_image(annotated_rgb, detections)
                last_annotated = cv2.cvtColor(
                    np.asarray(annotated_rgb),
                    cv2.COLOR_RGB2BGR,
                )

                if not frame_detections.empty:
                    for class_name in frame_detections["class_name"].tolist():
                        class_counts[str(class_name)] += 1
                    total_detections += len(frame_detections)

                processed += 1

            writer.write(last_annotated if last_annotated is not None else frame_bgr)
            frame_index += 1

            if progress is not None and total_frames:
                progress.progress(
                    min(frame_index / total_frames, 1.0),
                    text=f"Processing frame {frame_index:,} / {total_frames:,}",
                )
    finally:
        cap.release()
        writer.release()

    return {
        "frames_processed": processed,
        "total_detections": total_detections,
        "fps": fps,
        "class_counts": dict(class_counts),
        "total_frames": frame_index,
    }
