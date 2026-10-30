# SmartPark AI — GitHub-ready project

SmartPark AI is a Streamlit computer-vision prototype for local vehicle detection,
recorded-video analysis, and configurable parking-grid occupancy estimation.

The detector is **RF-DETR Small**, not Ultralytics YOLO. The app expects the
locally supplied PyTorch checkpoint at `models/weights.pt`.

## Project structure

```text
SmartPark_AI_GitHub_Ready_v2/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── PUT_MODEL_HERE.txt
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── parking.py
│   ├── analytics.py
│   └── video_processor.py
└── assets/
    ├── samples/
    │   └── .gitkeep
    └── outputs/
        └── .gitkeep
```

## Model checkpoint setup

1. Obtain the trained RF-DETR Small checkpoint from your Roboflow training run.
2. Copy the checkpoint into the project's `models/` directory.
3. The required local path is `models/weights.pt`.
4. Do not substitute an Ultralytics YOLO checkpoint.

The model weights are deliberately **not included** in this ZIP or intended for
GitHub. The `models/PUT_MODEL_HERE.txt` file explains where to place them.
The application displays a clear message when the checkpoint is missing.

The checkpoint is loaded locally with `RFDETRSmall` from the `rfdetr` package.
Inference does not call the hosted Roboflow prediction API and does not require
a Roboflow API key. The checkpoint must nevertheless be available on the
machine or deployment environment that performs inference.

The checkpoint inspected for this project did not embed the class-name string.
The application therefore maps its one foreground class ID (`0`) to
`Car_Topview`, based on the supplied training information.

## Windows setup

Use Python 3.10 or newer. A Python 3.11 virtual environment is a reasonable
starting point for this computer-vision stack.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CPU-first local test, the app initializes RF-DETR on CPU. PyTorch and
RF-DETR must be installed successfully before running the app.

## Run the application

From the project root:

```powershell
streamlit run app.py
```

## Features retained

- Streamlit dashboard and navigation.
- Local image detection with annotated bounding boxes.
- Detection table with class, confidence, coordinates, and center points.
- CSV export of image detections.
- Recorded-video processing and annotated MP4 download.
- Configurable rectangular-grid parking occupancy estimate.
- Analytics summary and class-count helper functions.
- Model status information and local-inference messaging.

## Important limitations

- The checkpoint is not included in the ZIP. Detection cannot run until the
  correct `models/weights.pt` file is supplied.
- The checkpoint metadata inspected for this project did not contain a saved
  `model_config` or class-name list. The model is instantiated as RF-DETR Small;
  the original training resolution/configuration should be checked against
  Roboflow's training configuration if available.
- The one-class label mapping `0: Car_Topview` comes from the project training
  information, not from a class-name string embedded in the checkpoint.
- The parking monitor divides the image into equal rectangular cells and
  assigns each detected object's center to a cell. This is an approximation,
  not calibrated parking-stall segmentation.
- Accurate stall occupancy requires a fixed camera and manually defined
  polygons for each parking bay, with perspective and occlusion testing.
- Video mode analyzes every Nth frame and reuses the last annotated frame
  between inference frames. Counts represent detections on analyzed frames,
  not unique vehicles.
- CPU inference is the conservative default. GPU/CUDA performance depends on
  the installed PyTorch build, NVIDIA driver, GPU support, resolution, and
  available VRAM.

## Streamlit Community Cloud

The source tree and Python imports are organized for a Streamlit deployment,
but deployment has **not** been tested on Streamlit Community Cloud.

The trained checkpoint is intentionally excluded from this repository ZIP.
A deployed instance must have access to the checkpoint at `models/weights.pt`
through an appropriate private deployment/storage arrangement. Do not commit
large model weights or credentials to a public GitHub repository.

## Troubleshooting

**Model not found:** place the actual RF-DETR checkpoint at
`models/weights.pt`. The placeholder text file is not a model.

**Model loading fails:** verify that the checkpoint is the RF-DETR Small
checkpoint from the intended training run and that the installed `rfdetr`
version is compatible with it. If available, retain the original
`training_config.json` for checking architecture settings.

**Import errors:** activate the project's virtual environment and run
`pip install -r requirements.txt` from the project root.

**Slow inference:** use CPU as the stable baseline, lower the inference image
size where appropriate, reduce video processing width, or increase frame
stride.

**Parking estimate is inaccurate:** the current grid is only a rectangular
approximation. Configure real stall polygons for a fixed camera before using
the output for operational parking decisions.
