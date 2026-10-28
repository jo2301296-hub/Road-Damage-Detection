# Road Damage Detection Using YOLOv8

A computer-vision project that detects possible road damage in images using a trained YOLOv8 object-detection model and a Gradio web interface.

## Features

- Upload a road image and visualize predicted bounding boxes.
- Adjust the confidence threshold.
- View predicted classes and confidence scores.
- Run locally or deploy on a compatible hosting service.

## Project structure

```text
Road-Damage-Detection/
├── app.py
├── requirements.txt
├── README.md
└── best.pt                 # Add your trained model weights; not included by default
```

## 1. Get the trained model

This repository does not include `best.pt` automatically because model weights are generated during training and may be large.

In your Kaggle notebook, the trained weights were saved at:

```text
/kaggle/working/road_damage/yolov8n_road_damage/weights/best.pt
```

Download that file from the notebook output, then place it in the same directory as `app.py`.

Alternatively, set `MODEL_PATH` to the location of your weights.

> **Important:** Do not commit large model files to GitHub if they exceed GitHub's file limits. Consider Git LFS or hosting the weights as a Kaggle Dataset/Hugging Face asset, and document the download instructions here.

## 2. Install dependencies

Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the app

Make sure `best.pt` is beside `app.py`, then run:

```bash
python app.py
```

Open the local URL printed by Gradio in your browser.

If the model is stored elsewhere, set `MODEL_PATH` before launching.

**Windows PowerShell:**
```powershell
$env:MODEL_PATH="C:\path\to\best.pt"
python app.py
```

**macOS/Linux:**
```bash
MODEL_PATH="/path/to/best.pt" python app.py
```

## Model information

- Model family: YOLOv8
- Task: object detection
- Input: road images
- Output: bounding boxes, predicted classes, and confidence scores

Reported evaluation results from the current notebook:

| Metric | Result |
|---|---:|
| Precision | 62.33% |
| Recall | 54.34% |
| mAP@50 | 57.62% |
| mAP@50–95 | 30.77% |

These values describe the evaluated model run and can vary with the dataset split, model weights, and software version. They are not a guarantee of performance on real-world images.

## Class labels

The app currently maps class IDs as follows:

| Class ID | Display label |
|---:|---|
| 0 | alligator_crack |
| 1 | transverse_crack |
| 2 | longitudinal_crack |
| 3 | other_corruption |
| 4 | pothole |

**Verify this mapping against the official RDD-2022 dataset configuration before publishing results.** If the official mapping differs, update `CLASS_NAME_OVERRIDES` in `app.py`.

## Dataset and notebook

- Kaggle notebook: https://www.kaggle.com/code/youssef223rlkhi9tj/rdd-2022
- GitHub repository: https://github.com/jo2301296-hub/Road-Damage-Detection.git

## Notes

- A prediction of “no damage detected” does not prove that a road is damage-free.
- Use predictions as decision support, not as a substitute for professional road inspection.
- Keep dataset licensing and attribution requirements in mind when redistributing data or model weights.

## License

Add a license appropriate for your code and verify the dataset/model licensing terms before redistribution.
