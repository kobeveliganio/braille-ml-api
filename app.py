from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import base64
import os

app = Flask(__name__)

# ✅ Allow only your production frontend domain
CORS(app, resources={r"/*": {"origins": "https://smartvision-betl.onrender.com"}})

# Load YOLO model once
model = YOLO("best.pt")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Run YOLO detection
    results = model.predict(image, save=True)
    result_image_path = results[0].save_dir + "/result.jpg"

    # Convert result image to base64
    with open(result_image_path, "rb") as img_file:
        img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify({"result_image": img_b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
