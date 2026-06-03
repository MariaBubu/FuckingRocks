#!/usr/bin/env python3
"""
Fossil Web App — Fast inference using ONNX Runtime (no PyTorch import needed).

Architecture:
  - Uses onnxruntime for ResNet18 inference (~1s import vs PyTorch's 477s).
  - Uses OpenCV for bounding box detection (contour-based, no ML model needed).
  - Server starts instantly and classification works within seconds.
"""
from __future__ import annotations

import os

# Force offline modes
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Force single-threaded to prevent deadlocks on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import base64
import html
import io
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data/crop_classifier_controlled_aug/model"
MODEL_PATH = MODEL_DIR / "best_resnet18.pt"
ONNX_PATH = MODEL_DIR / "best_resnet18.onnx"
CLASSES_PATH = MODEL_DIR / "classes.json"
YOLO_BOX_MODEL_PATH = (
    ROOT
    / "runs/detect/outputs/box_training/yolo11n_fossil_first_run/weights/best.pt"
)
PORT = 5050
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

CLASS_DISPLAY = {
    "Coral": {
        "label": "Coral fossil",
        "short": "Coral",
        "description": "The model thinks this cropped fossil is closer to the coral examples it saw during training.",
    },
    "Shell": {
        "label": "Brachiopod fossil",
        "short": "Brachiopod",
        "description": "The model thinks this cropped fossil is closer to the brachiopod/shell examples it saw during training.",
    },
    "FuckingCorals": {
        "label": "Coral fossil",
        "short": "Coral",
        "description": "The model thinks this image is closer to the coral examples it saw during training.",
    },
    "FuckingShells": {
        "label": "Brachiopod fossil",
        "short": "Brachiopod",
        "description": "The model thinks this image is closer to the brachiopod examples it saw during training.",
    },
}

# Model globals
MODEL_SESSION = None
MODEL_BACKEND = ""
CLASSES: list[str] = []
MODEL_STATE = "idle"
MODEL_ERROR = ""
MODEL_LOCK = threading.Lock()
YOLO_MODEL = None
YOLO_STATE = "idle"
YOLO_ERROR = ""
YOLO_LOCK = threading.Lock()


def load_model_once():
    global MODEL_SESSION, MODEL_BACKEND, CLASSES, MODEL_STATE, MODEL_ERROR
    if MODEL_SESSION is not None:
        return MODEL_SESSION, CLASSES

    with MODEL_LOCK:
        if MODEL_SESSION is not None:
            return MODEL_SESSION, CLASSES

        MODEL_STATE = "loading"
        MODEL_ERROR = ""
        try:
            if ONNX_PATH.exists() and CLASSES_PATH.exists():
                import onnxruntime as ort

                with open(CLASSES_PATH) as f:
                    CLASSES = json.load(f)

                t0 = time.time()
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(
                    str(ONNX_PATH),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                elapsed = time.time() - t0
                MODEL_SESSION = session
                MODEL_BACKEND = "onnx"
                MODEL_STATE = "ready"
                print(f"  ✓ ONNX model loaded in {elapsed:.2f}s (classes: {CLASSES})", flush=True)
                return MODEL_SESSION, CLASSES

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Could not find model at {MODEL_PATH}"
                )

            t0 = time.time()
            from numpy_resnet18 import NumpyResNet18

            session = NumpyResNet18(MODEL_PATH)
            CLASSES = session.classes
            if not CLASSES_PATH.exists():
                CLASSES_PATH.write_text(json.dumps(CLASSES), encoding="utf-8")
            elapsed = time.time() - t0
            MODEL_SESSION = session
            MODEL_BACKEND = "numpy"
            MODEL_STATE = "ready"
            print(f"  ✓ NumPy ResNet model loaded in {elapsed:.2f}s (classes: {CLASSES})", flush=True)
        except Exception as exc:
            MODEL_STATE = "error"
            MODEL_ERROR = str(exc)
            print(f"  ✗ Model load failed: {exc}", flush=True)
            raise
    return MODEL_SESSION, CLASSES


def model_status() -> dict:
    if MODEL_SESSION is not None:
        backend = "ONNX" if MODEL_BACKEND == "onnx" else "NumPy"
        return {"state": "ready", "message": f"{backend} model ready"}
    if ONNX_PATH.exists() and CLASSES_PATH.exists():
        return {"state": "idle", "message": "ONNX model will load on first use"}
    if MODEL_PATH.exists():
        return {
            "state": "idle",
            "message": "NumPy fallback will load on first use",
        }
    if MODEL_STATE == "loading":
        return {"state": "loading", "message": "Model loading..."}
    if MODEL_STATE == "error":
        return {"state": "error", "message": MODEL_ERROR or "Model failed to load"}
    return {"state": "setup", "message": "Model file missing"}


def preload_model_background():
    if not (ONNX_PATH.exists() or MODEL_PATH.exists()):
        return

    def preload():
        try:
            load_model_once()
        except Exception as exc:
            print(f"  ⚠ Background model preload failed: {exc}", flush=True)

    threading.Thread(target=preload, daemon=True).start()


def preload_yolo_background():
    if not YOLO_BOX_MODEL_PATH.exists():
        return

    def preload():
        try:
            load_yolo_once()
        except Exception as exc:
            print(f"  ⚠ Background YOLO preload failed: {exc}", flush=True)

    threading.Thread(target=preload, daemon=True).start()


def image_to_numpy(image: Image.Image):
    """Convert PIL image to normalized numpy array (1, 3, 224, 224) float32."""
    import numpy as np
    from PIL import Image

    image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    # Convert to numpy: (224, 224, 3) uint8
    arr = np.array(image, dtype=np.float32) / 255.0
    # Transpose to (3, 224, 224) and normalize with ImageNet stats
    arr = arr.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    # Add batch dimension: (1, 3, 224, 224)
    return arr[np.newaxis, ...]


def softmax(x):
    """Compute softmax along the last axis."""
    import numpy as np

    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def preview_data_url(image: Image.Image) -> str:
    from PIL import Image

    preview = image.convert("RGB")
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def classify_image(image: Image.Image) -> dict:
    session, classes = load_model_once()
    input_data = image_to_numpy(image)

    if MODEL_BACKEND == "onnx":
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]  # shape: (1, num_classes)
    else:
        logits = session.predict_logits(input_data)
    probabilities = softmax(logits).squeeze(0)  # shape: (num_classes,)

    ranked = sorted(
        [
            {
                "class_name": class_name,
                "display": CLASS_DISPLAY[class_name]["short"],
                "confidence": float(probabilities[index]),
            }
            for index, class_name in enumerate(classes)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )
    top = ranked[0]
    display = CLASS_DISPLAY[top["class_name"]]
    return {
        "label": display["label"],
        "description": display["description"],
        "confidence_percent": round(top["confidence"] * 100, 1),
        "ranked": ranked,
    }


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0
    return intersection_area / union_area


def apply_nms(detections, iou_threshold=0.3):
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for det in sorted_dets:
        box = [det["box"]["xmin"], det["box"]["ymin"], det["box"]["xmax"], det["box"]["ymax"]]
        overlap = False
        for kept_det in kept:
            kept_box = kept_det["bbox"]
            if calculate_iou(box, kept_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            det["bbox"] = box
            kept.append(det)
    return kept


def load_yolo_once():
    global YOLO_MODEL, YOLO_STATE, YOLO_ERROR

    if YOLO_MODEL is not None:
        return YOLO_MODEL

    with YOLO_LOCK:
        if YOLO_MODEL is not None:
            return YOLO_MODEL

        if not YOLO_BOX_MODEL_PATH.exists():
            YOLO_STATE = "error"
            YOLO_ERROR = f"YOLO box model not found: {YOLO_BOX_MODEL_PATH}"
            raise FileNotFoundError(YOLO_ERROR)

        YOLO_STATE = "loading"
        YOLO_ERROR = ""
        start = time.time()
        try:
            from ultralytics import YOLO

            YOLO_MODEL = YOLO(str(YOLO_BOX_MODEL_PATH))
            YOLO_STATE = "ready"
            print(f"  ✓ YOLO fossil box model loaded in {time.time() - start:.2f}s", flush=True)
            return YOLO_MODEL
        except Exception as exc:
            YOLO_STATE = "error"
            YOLO_ERROR = str(exc)
            raise


def intersection_area(box_a: list[int], box_b: list[int]) -> int:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def remove_nested_detections(detections: list[dict], containment_threshold: float = 0.85) -> list[dict]:
    kept: list[dict] = []
    for detection in sorted(detections, key=lambda item: (item["area"], item["score"]), reverse=True):
        nested = False
        for larger in kept:
            if larger["area"] <= detection["area"]:
                continue
            overlap = intersection_area(detection["bbox"], larger["bbox"])
            if detection["area"] and overlap / detection["area"] >= containment_threshold:
                nested = True
                break
        if not nested:
            kept.append(detection)
    return sorted(kept, key=lambda item: item["score"], reverse=True)


def detect_and_classify_multiple_fossils(image: Image.Image, confidence: float = 0.25) -> dict:
    from PIL import ImageDraw, ImageFont

    confidence = max(0.01, min(0.95, confidence))
    yolo_model = load_yolo_once()
    yolo_results = yolo_model.predict(
        source=image.convert("RGB"),
        imgsz=640,
        conf=confidence,
        iou=0.45,
        max_det=40,
        verbose=False,
        device="cpu",
    )
    boxes = yolo_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return {
            "success": True,
            "no_detections": True,
            "confidence_threshold": confidence,
            "message": f"No fossil boxes found above the {confidence:.2f} confidence threshold.",
            "fossils": [],
            "image_width": image.width,
            "image_height": image.height,
            "image_url": preview_data_url(image),
            "annotated_image_url": preview_data_url(image),
        }

    detections = []
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(round(value)) for value in box.xyxy[0].tolist()]
        xmin = max(0, min(image.width - 1, xmin))
        ymin = max(0, min(image.height - 1, ymin))
        xmax = max(xmin + 1, min(image.width, xmax))
        ymax = max(ymin + 1, min(image.height, ymax))
        score = float(box.conf[0].item())
        area = (xmax - xmin) * (ymax - ymin)
        detections.append({"bbox": [xmin, ymin, xmax, ymax], "score": score, "area": area})

    detections = remove_nested_detections(detections)[:24]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(7, round(min(image.width, image.height) / 95))
    font_size = max(24, round(min(image.width, image.height) / 32))
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
    except Exception:
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            label_font = ImageFont.load_default()

    fossils_list = []
    for index, det in enumerate(detections):
        fossil_id = index + 1
        box = det["bbox"]
        xmin, ymin, xmax, ymax = box

        cropped = image.crop((xmin, ymin, xmax, ymax))
        try:
            classification = classify_image(cropped)
            classification_short = classification["ranked"][0]["display"]
            classification_confidence = classification["confidence_percent"]
        except Exception as exc:
            classification = {
                "label": "Classification unavailable",
                "description": f"Could not classify this crop: {exc}",
                "confidence_percent": 0,
                "ranked": [],
            }
            classification_short = "Unclassified"
            classification_confidence = 0
        color = "#00ff5a"
        shadow_color = "#001b0b"

        for offset in range(line_width + 2, line_width + 6):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=shadow_color
            )
        for offset in range(line_width):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=color
            )

        try:
            label = f"Fossil {fossil_id}  {classification_short} {classification_confidence:.0f}%"
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            pad_x = max(8, line_width)
            pad_y = max(5, line_width // 2)
            label_x1 = xmin
            label_y1 = max(0, ymin - label_height - (pad_y * 2) - line_width)
            label_x2 = min(image.width, label_x1 + label_width + (pad_x * 2))
            label_y2 = min(image.height, label_y1 + label_height + (pad_y * 2))
            draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)
            draw.text((label_x1 + pad_x, label_y1 + pad_y), label, fill="#001b0b", font=label_font)
        except Exception:
            pass

        fossils_list.append({
            "id": fossil_id,
            "score": float(det["score"]),
            "bbox": [xmin, ymin, xmax, ymax],
            "classification": classification,
            "cropped_image_url": preview_data_url(cropped)
        })

    return {
        "success": True,
        "confidence_threshold": confidence,
        "image_width": image.width,
        "image_height": image.height,
        "image_url": preview_data_url(image),
        "fossils": fossils_list,
        "annotated_image_url": preview_data_url(annotated)
    }


def render_page(result=None, image_url=None, error=None) -> bytes:
    template = Template((ROOT / "templates" / "index.html").read_text(encoding="utf-8"))
    result_html = render_result(result, image_url) if result else render_empty_state()
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    page = template.safe_substitute(
        stylesheet_url="/static/styles.css",
        error_html=error_html,
        result_html=result_html,
        model_name="original + enhanced ResNet18",
        model_path="data/preprocessing_experiments/original_enhanced/model/best_resnet18.pt",
        model_status_message=html.escape(model_status()["message"]),
    )
    return page.encode("utf-8")


def render_empty_state() -> str:
    return """
    <div class="empty-state">
      <div class="sample-tile"></div>
      <h2>Upload an image to analyze it.</h2>
      <p>The app will draw fossil boxes, then classify every detected crop as coral or brachiopod.</p>
    </div>
    """


def render_result(result: dict, image_url: str) -> str:
    bars = []
    for item in result["ranked"]:
        confidence = item["confidence"] * 100
        bars.append(
            f"""
            <div class="bar-row bar-row-{html.escape(item["class_name"])}">
              <div class="bar-label">
                <span>{html.escape(item["display"])}</span>
                <span>{confidence:.1f}%</span>
              </div>
              <div class="bar-track">
                <div class="bar-fill" style="width: {confidence}%"></div>
              </div>
            </div>
            """
        )

    return f"""
    <div class="result-grid">
      <div class="preview-wrap">
        <button id="clear-result-btn" class="clear-result-btn" type="button" title="Remove image and reset">
          <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
        <img src="{image_url}" alt="Uploaded fossil photo">
      </div>
      <div class="prediction">
        <div class="prediction-info">
          <p class="eyebrow">Analysis Result</p>
          <h2>{html.escape(result["label"])}</h2>
          <p class="description">{html.escape(result["description"])}</p>
        </div>
        <div class="confidence-ring" title="Overall Model Confidence Score">
          <span class="ring-value">{result["confidence_percent"]}%</span>
          <span class="ring-label">Certainty</span>
        </div>
        <div class="bars">
          <p class="bars-heading">Probability Breakdown</p>
          {''.join(bars)}
        </div>
      </div>
    </div>
    """


def _parse_multipart_form(handler):
    """Parse multipart form data without the deprecated cgi module."""
    content_type = handler.headers.get("Content-Type", "")
    if "boundary=" not in content_type:
        return {}, {}

    boundary = content_type.split("boundary=")[1].strip()
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    content_length = int(handler.headers.get("Content-Length", "0"))
    body = handler.rfile.read(content_length)

    boundary_bytes = boundary.encode("utf-8")
    parts = body.split(b"--" + boundary_bytes)
    files = {}
    fields = {}

    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        headers_section = part[:header_end].decode("utf-8", errors="replace")
        content = part[header_end + 4:]
        if content.endswith(b"\r\n"):
            content = content[:-2]

        if 'name="' not in headers_section:
            continue
        field_name = headers_section.split('name="')[1].split('"')[0]
        if 'filename="' in headers_section:
            filename = headers_section.split('filename="')[1].split('"')[0]
            files[field_name] = (content, filename)
        else:
            fields[field_name] = content.decode("utf-8", errors="replace")

    return files, fields


def _parse_multipart(handler):
    files, _fields = _parse_multipart_form(handler)
    return files.get("image", (None, None))


class FossilHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self.send_html(render_page())
        elif path == "/model-status":
            self.send_json(model_status())
        elif path.startswith("/static/"):
            self.send_static(path)
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))

        if self.path == "/detect":
            if content_length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "That image is too large. Limit is 20 MB."})
                return
            try:
                files, fields = _parse_multipart_form(self)
                image_bytes, filename = files.get("image", (None, None))
                if image_bytes is None or not filename:
                    self.send_json({"error": "Please select an image file."})
                    return
                try:
                    confidence = float(fields.get("confidence", "0.25"))
                except ValueError:
                    confidence = 0.25
                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = detect_and_classify_multiple_fossils(image, confidence=confidence)
                self.send_json(result)
            except Exception as exc:
                self.send_json({"success": False, "error": str(exc)})
            return

        if self.path == "/classify":
            if content_length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "That image is too large. Limit is 20 MB."})
                return
            try:
                image_bytes, filename = _parse_multipart(self)
                if image_bytes is None or not filename:
                    self.send_json({"error": "Please select an image file."})
                    return
                from PIL import Image

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result = classify_image(image)
                self.send_json({
                    "success": True,
                    "result": result,
                    "image_url": preview_data_url(image)
                })
            except Exception as exc:
                self.send_json({"success": False, "error": str(exc)})
            return

        # Original fallback form action
        if self.path != "/":
            self.send_error(404)
            return

        if content_length > MAX_UPLOAD_BYTES:
            self.send_html(render_page(error="That image is too large. Try a file under 20 MB."))
            return

        try:
            image_bytes, filename = _parse_multipart(self)
            if image_bytes is None or not filename:
                self.send_html(render_page(error="Choose an image first."))
                return
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            result = classify_image(image)
            self.send_html(render_page(result=result, image_url=preview_data_url(image)))
        except Exception as exc:
            self.send_html(render_page(error=f"Could not classify that image: {exc}"))

    def send_html(self, content: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_static(self, path: str):
        static_path = (ROOT / path.lstrip("/")).resolve()
        if ROOT not in static_path.parents or not static_path.exists():
            self.send_error(404)
            return
        content = static_path.read_bytes()
        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    print("=" * 60, flush=True)
    print("🚀 Fossil Web App — ONNX Runtime Mode", flush=True)
    print("   No PyTorch import during website startup.", flush=True)
    print("=" * 60, flush=True)

    if ONNX_PATH.exists() and CLASSES_PATH.exists():
        print("  ✓ Fast ONNX model found; it will load on first classify.", flush=True)
    elif MODEL_PATH.exists():
        print("  ✓ PyTorch .pt model found; NumPy fallback will load on first classify.", flush=True)
    else:
        print("  ⚠ No model file found; classification will show an error.", flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", PORT), FossilHandler)
    print(f"✅ Server is live at http://127.0.0.1:{PORT}/", flush=True)
    print("   Models are warming in the background; page stays usable.", flush=True)
    print(f"   Press Ctrl+C to stop.", flush=True)
    print("=" * 60, flush=True)
    preload_model_background()
    preload_yolo_background()
    server.serve_forever()


if __name__ == "__main__":
    main()
