#!/usr/bin/env python3
from __future__ import annotations

import base64
import cgi
import html
import io
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import unquote

from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data/preprocessing_experiments/original_enhanced/model/best_resnet18.pt"
PORT = 5050
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

CLASS_DISPLAY = {
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

MODEL = None
CLASSES: list[str] = []
MODEL_STATE = "idle"
MODEL_ERROR = ""
MODEL_LOCK = threading.Lock()


def load_model_once():
    global MODEL, CLASSES, MODEL_STATE, MODEL_ERROR
    if MODEL is not None:
        return MODEL, CLASSES

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, CLASSES

        MODEL_STATE = "loading"
        MODEL_ERROR = ""
        try:
            import torch
            from evaluate_wild_no_torchvision import ResNet18

            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
            CLASSES = checkpoint["classes"]
            model = ResNet18(num_classes=len(CLASSES))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            MODEL = model
            MODEL_STATE = "ready"
        except Exception as exc:
            MODEL_STATE = "error"
            MODEL_ERROR = str(exc)
            raise
    return MODEL, CLASSES


def preload_model_background():
    def preload():
        try:
            load_model_once()
        except Exception as exc:
            print(f"Model preload failed: {exc}", flush=True)

    thread = threading.Thread(target=preload, daemon=True)
    thread.start()


def model_status() -> dict:
    if MODEL is not None:
        return {"state": "ready", "message": "Model ready"}
    if MODEL_STATE == "loading":
        return {"state": "loading", "message": "Model loading"}
    if MODEL_STATE == "error":
        return {"state": "error", "message": MODEL_ERROR or "Model failed to load"}
    return {"state": "idle", "message": "Model warming up"}


def image_to_tensor(image: Image.Image):
    import torch

    image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(224, 224, 3).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def preview_data_url(image: Image.Image) -> str:
    preview = image.convert("RGB")
    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def classify_image(image: Image.Image) -> dict:
    import torch

    model, classes = load_model_once()
    tensor = image_to_tensor(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)

    ranked = sorted(
        [
            {
                "class_name": class_name,
                "display": CLASS_DISPLAY[class_name]["short"],
                "confidence": float(probabilities[index].item()),
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
      <h2>Upload an image to classify it.</h2>
      <p>The app will run the current best local model and return a coral or brachiopod prediction with confidence.</p>
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
        # 1. New dynamic multiple-image classification endpoint
        if self.path == "/classify":
            if int(self.headers.get("Content-Length", "0")) > MAX_UPLOAD_BYTES:
                self.send_json({"error": "That image is too large. Limit is 20 MB."})
                return

            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                field = form["image"] if "image" in form else None
                if field is None or not getattr(field, "filename", ""):
                    self.send_json({"error": "Please select an image file."})
                    return

                image_bytes = field.file.read()
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

        # 2. Original fallback form action
        if self.path != "/":
            self.send_error(404)
            return

        if int(self.headers.get("Content-Length", "0")) > MAX_UPLOAD_BYTES:
            self.send_html(render_page(error="That image is too large. Try a file under 20 MB."))
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            field = form["image"] if "image" in form else None
            if field is None or not getattr(field, "filename", ""):
                self.send_html(render_page(error="Choose an image first."))
                return

            image_bytes = field.file.read()
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
    server = ThreadingHTTPServer(("127.0.0.1", PORT), FossilHandler)
    print(f"Fossil classifier running at http://127.0.0.1:{PORT}/", flush=True)
    print("Model is warming up in the background.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    preload_model_background()
    server.serve_forever()


if __name__ == "__main__":
    main()
