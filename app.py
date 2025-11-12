from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import io
from PIL import Image
import base64

app = Flask(__name__)

# Allowed origins (your frontend)
ALLOWED_ORIGINS = ["https://smartvision-betl.onrender.com", "http://localhost:3000"]

# Basic CORS init (helps, but we'll enforce headers in after_request too)
CORS(app,
     resources={r"/*": {"origins": ALLOWED_ORIGINS}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

# Optional API key
ML_API_KEY = os.environ.get("ML_API_KEY", "my-secret-key-123")

# Lazy-load YOLO
model = None
def get_model():
    global model
    if model is None:
        try:
            model = YOLO("best.pt")
            print("✅ YOLO model loaded successfully.")
        except Exception as e:
            print("❌ Failed to load YOLO model:", e)
            model = None
    return model

# Ensure CORS headers always present (and only echo allowed origin)
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response

# Make sure OPTIONS is handled even if something else intercepts it
@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    resp = make_response()
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return resp, 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_instance = get_model()
        if model_instance is None:
            return jsonify({"error": "YOLO model not loaded"}), 500

        # Validate API key header (optional)
        api_key = request.headers.get("Authorization")
        if not api_key or api_key != f"Bearer {ML_API_KEY}":
            return jsonify({"error": "Unauthorized: Invalid API key"}), 401

        # Receive file
        file = request.files.get("file") or request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Convert to numpy BGR for YOLO/cv2
        image_stream = io.BytesIO(file.read())
        pil_image = Image.open(image_stream).convert("RGB")
        img_np = np.array(pil_image)[:, :, ::-1].copy()

        # Run prediction
        results = model_instance.predict(source=img_np, conf=0.25, verbose=False)

        # Annotate image
        annotated_img = results[0].plot()

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        predictions = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = results[0].names[cls_id]
            predictions.append({"label": label, "confidence": round(confidence, 2)})

        translated_text = "".join([p["label"][0].upper() for p in predictions])

        response = jsonify({
            "annotated_image_base64": img_b64,
            "braille_text": [p["label"] for p in predictions],
            "translated_text": translated_text
        })

        # after_request will add CORS headers
        return response, 200

    except Exception as e:
        print("🔥 Server error:", e)
        response = jsonify({"error": str(e)})
        return response, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port)
