from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import base64
import os

app = Flask(__name__)

# ✅ Allow requests from your frontend domain only (better security)
CORS(app, origins=["https://smartvision-betl.onrender.com"])  # Replace with your deployed domain

# ✅ Load YOLO model once
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
        # ✅ Check if model is loaded
        if model is None:
            return jsonify({"error": "YOLO model not loaded"}), 500

        # ✅ Check for image in request
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # ✅ Convert file to RGB image
        try:
            image = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image format"}), 400

        # ✅ Run YOLO prediction
        results = model.predict(image, save=True)
        if not results or not results[0].save_dir:
            return jsonify({"error": "YOLO failed to produce results"}), 500

        result_image_path = os.path.join(results[0].save_dir, "result.jpg")

        # ✅ Check if result exists
        if not os.path.exists(result_image_path):
            return jsonify({"error": "Result image not found"}), 500

        # ✅ Encode result image as base64
        with open(result_image_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return jsonify({"result_image": img_b64}), 200

    except Exception as e:
        # ✅ Return a JSON error instead of crashing
        print("🔥 Server error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
