import numpy as np
from PIL import Image, ImageDraw, ImageFont

def analyze_grid(detections, image_width: int, image_height: int, rows: int, cols: int):
    """Approximate occupancy by assigning each detected object's center to a grid cell."""
    occupied_cells = set()
    if not detections.empty:
        for _, item in detections.iterrows():
            x = float(item["center_x"])
            y = float(item["center_y"])
            col = min(cols - 1, max(0, int(x / max(image_width, 1) * cols)))
            row = min(rows - 1, max(0, int(y / max(image_height, 1) * rows)))
            occupied_cells.add((row, col))
    slot_rows = []
    for r in range(rows):
        states = []
        for c in range(cols):
            states.append("Occupied" if (r, c) in occupied_cells else "Available")
        slot_rows.append({"Row": f"Row {r+1}", **{f"Slot {c+1}": s for c, s in enumerate(states)}})
    total = rows * cols
    occupied = len(occupied_cells)
    return {
        "total": total,
        "occupied": occupied,
        "available": total - occupied,
        "occupancy_rate": (occupied / total * 100) if total else 0,
        "occupied_cells": occupied_cells,
        "slot_rows": slot_rows,
    }

def draw_grid(image, occupancy, rows: int, cols: int):
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    w, h = canvas.size
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c*w/cols), int(r*h/rows)
            x2, y2 = int((c+1)*w/cols), int((r+1)*h/rows)
            is_occupied = (r,c) in occupancy["occupied_cells"]
            fill = (239, 68, 68, 55) if is_occupied else (34, 197, 94, 45)
            outline = (248, 113, 113, 230) if is_occupied else (74, 222, 128, 230)
            draw.rectangle((x1,y1,x2,y2), fill=fill, outline=outline, width=max(2, w//500))
            label = f"R{r+1}-S{c+1}  {'FULL' if is_occupied else 'FREE'}"
            draw.text((x1+6,y1+6), label, fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,210))
    return canvas
