from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rfdetr import RFDETRSmall


CLASS_NAMES = {0: "Car_Topview"}
DETECTION_COLUMNS = [
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


def load_detector(model_path: str, device: str = "cpu"):
    """Load the locally saved RF-DETR Small checkpoint.

    trust_checkpoint=True is used because this project loads the checkpoint
    supplied by its owner. Only load checkpoint files from a trusted source.
    """
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"RF-DETR model weights were not found: {path}. "
            "Place the trained checkpoint at models/weights.pt."
        )

    return RFDETRSmall(
        pretrain_weights=str(path.resolve()),
        num_classes=1,
        device=device,
        trust_checkpoint=True,
    )


def _class_name(class_id: int) -> str:
    """Resolve the single trained SmartPark class."""
    return CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")


def detections_to_frame(detections) -> pd.DataFrame:
    """Convert RF-DETR/Supervision detections to SmartPark's tabular schema."""
    rows = []
    if detections is not None and detections.xyxy is not None:
        boxes = detections.xyxy
        confidences = detections.confidence
        class_ids = detections.class_id

        for index, box in enumerate(boxes):
            if confidences is None or class_ids is None:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            class_id = int(class_ids[index])
            score = float(confidences[index])
            rows.append(
                {
                    "class_name": _class_name(class_id),
                    "confidence": round(score, 4),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "center_x": round((x1 + x2) / 2, 1),
                    "center_y": round((y1 + y2) / 2, 1),
                }
            )

    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def annotate_image(image: Image.Image, detections) -> Image.Image:
    """Draw RF-DETR bounding boxes on a copy of an RGB PIL image."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    rows = detections_to_frame(detections)

    for row in rows.to_dict(orient="records"):
        x1, y1 = int(row["x1"]), int(row["y1"])
        x2, y2 = int(row["x2"]), int(row["y2"])
        label = f'{row["class_name"]} {row["confidence"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 160), width=3)
        text_box = draw.textbbox((x1, max(0, y1 - 18)), label)
        draw.rectangle(text_box, fill=(0, 80, 70))
        draw.text((x1, max(0, y1 - 18)), label, fill=(255, 255, 255))

    return annotated


def detect_image(model, image, conf: float = 0.35, imgsz: int = 640):
    """Run local RF-DETR inference on a PIL RGB image.

    RF-DETR's `shape` is an (height, width) tuple. The selected image size
    must satisfy the divisibility requirements of the installed RF-DETR
    variant (the UI choices in app.py are multiples of 32).
    """
    rgb_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image)
    detections = model.predict(
        images=rgb_image,
        threshold=float(conf),
        shape=(int(imgsz), int(imgsz)),
    )
    annotated_rgb = annotate_image(rgb_image, detections)
    return annotated_rgb, detections_to_frame(detections)


def model_info(model_path: str):
    path = Path(model_path)
    exists = path.is_file()
    return {
        "model_path": str(path),
        "weights_found": exists,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2) if exists else None,
        "model_architecture": "RF-DETR Small",
        "checkpoint_format": "PyTorch checkpoint (.pt)",
        "inference_mode": "Local / on-device",
        "hosted_api_required": False,
        "expected_class": "Car_Topview",
    }
