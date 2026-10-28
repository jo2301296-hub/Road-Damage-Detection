import os
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "best.pt"))
CONFIDENCE_DEFAULT = float(os.getenv("CONFIDENCE_DEFAULT", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

CLASS_NAME_OVERRIDES = {
    # Verify these names against the dataset's official class mapping before publishing.
    0: "alligator_crack",
    1: "transverse_crack",
    2: "longitudinal_crack",
    3: "other_corruption",
    4: "pothole",
}


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}\n"
            "Place best.pt beside app.py, or set the MODEL_PATH environment variable."
        )
    return YOLO(str(MODEL_PATH))


model = load_model()


def predict_road_damage(image, confidence):
    if image is None:
        return None, "Upload a road image to begin."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = model.predict(
        source=image_bgr,
        conf=float(confidence),
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]

    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        report = (
            "No damage detected at this confidence threshold.\n"
            "Try lowering the confidence threshold. This does not guarantee the road is damage-free."
        )
        return annotated, report

    lines = []
    counts = {}
    for box in boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        class_name = CLASS_NAME_OVERRIDES.get(
            class_id, model.names.get(class_id, f"class_{class_id}")
        )
        counts[class_name] = counts.get(class_name, 0) + 1
        lines.append(f"- {class_name}: {score:.1%} confidence")

    summary = "\n".join(f"- {name}: {count}" for name, count in sorted(counts.items()))
    report = (
        f"Detections: {len(boxes)}\n\n"
        f"Counts by class:\n{summary}\n\n"
        f"Detection details:\n" + "\n".join(lines)
    )
    return annotated, report


with gr.Blocks(title="Road Damage Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛣️ Road Damage Detection
        Detect possible road damage in an image using a trained YOLOv8 model.

        Upload a road image, choose the confidence threshold, and run detection.
        Predictions are model estimates and should be reviewed by a human.
        """
    )

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Road image")
        output_image = gr.Image(type="numpy", label="Detection result")

    confidence_slider = gr.Slider(
        minimum=0.10,
        maximum=0.90,
        value=CONFIDENCE_DEFAULT,
        step=0.05,
        label="Confidence threshold",
    )
    detect_button = gr.Button("Detect road damage", variant="primary")
    report_output = gr.Textbox(label="Detection report", lines=12)

    detect_button.click(
        fn=predict_road_damage,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, report_output],
    )

if __name__ == "__main__":
    demo.launch()
