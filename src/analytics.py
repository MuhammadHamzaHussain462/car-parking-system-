import pandas as pd

def build_summary(detections: pd.DataFrame):
    if detections is None or detections.empty:
        return {"total_detections": 0, "unique_classes": 0, "average_confidence": 0.0}
    return {
        "total_detections": int(len(detections)),
        "unique_classes": int(detections["class_name"].nunique()),
        "average_confidence": float(detections["confidence"].mean()),
    }

def class_count_frame(detections: pd.DataFrame):
    if detections is None or detections.empty:
        return pd.DataFrame(columns=["class_name","count"])
    return (detections.groupby("class_name").size().reset_index(name="count")
            .sort_values("count", ascending=False))
