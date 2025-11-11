from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import os
import io
from PIL import Image
import base64

# Initialize app
app = Flask(__name__)

# ✅ Allow CORS only from your React frontend
CORS(app, origins=["https://smartvision-betl.onrender.com"], supports_credentials=True)

# Optional: Set your API key for requests
ML_API_KEY = os.environ.get("ML_API_KEY", "my-secret-key-123")  # set in Render .env

# Load YOLO model once
try:
    model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print("❌ Failed to load YOLO model:", e)
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "YOLO model not loaded"}), 500

        # Optional: Validate API key
        api_key = request.headers.get("Authorization")
        if not api_key or api_key != f"Bearer {ML_API_KEY}":
            return jsonify({"error": "Unauthorized: Invalid API key"}), 401

        # Check uploaded file
        file = request.files.get("file") or request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Convert to OpenCV image
        image_stream = io.BytesIO(file.read())
        pil_image = Image.open(image_stream).convert("RGB")
        img_np = np.array(pil_image)[:, :, ::-1].copy()  # RGB → BGR

        # Run YOLO prediction
        results = model.predict(source=img_np, conf=0.25, verbose=False)

        # Annotate image
        annotated_img = results[0].plot()  # numpy array

        # Encode annotated image to base64
        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Build predictions array
        predictions = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = results[0].names[cls_id]
            predictions.append({
                "label": label,
                "confidence": round(confidence, 2)
            })

        # Simple translation example
        translated_text = "".join([p["label"][0].upper() for p in predictions])

        return jsonify({
            "annotated_image_base64": img_b64,
            "braille_text": [p["label"] for p in predictions],
            "translated_text": translated_text
        }), 200

    except Exception as e:
        print("🔥 Server error:", e)
        return jsonify({"error": str(e)}), 500

# Keep Flask runnable locally
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
