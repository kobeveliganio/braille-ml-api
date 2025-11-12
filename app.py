from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2, numpy as np, os, io, base64
from PIL import Image

app = Flask(__name__)

# ✅ Allow only your deployed frontend (no duplicates)
CORS(
    app,
    origins=["https://smartvision-betl.onrender.com"],
    supports_credentials=True
)

# Lazy-load YOLO model
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

# API Key
ML_API_KEY = os.environ.get("ML_API_KEY", "my-secret-key-123")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_instance = get_model()
        if model_instance is None:
            return jsonify({"error": "YOLO model not loaded"}), 500

        # Check API key
        api_key = request.headers.get("Authorization")
        if not api_key or api_key != f"Bearer {ML_API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401

        # Get uploaded file
        file = request.files.get("file") or request.files.get("image")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Convert to OpenCV
        pil_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_np = np.array(pil_image)[:, :, ::-1].copy()

        # YOLO prediction
        results = model_instance.predict(source=img_np, conf=0.25, verbose=False)
        annotated_img = results[0].plot()

        _, buffer = cv2.imencode(".jpg", annotated_img)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        predictions = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            predictions.append({"label": label, "confidence": float(box.conf[0])})

        translated_text = "".join([p["label"][0].upper() for p in predictions])

        return jsonify({
            "annotated_image_base64": img_b64,
            "braille_text": [p["label"] for p in predictions],
            "translated_text": translated_text
        }), 200

    except Exception as e:
        print("🔥 Server error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
