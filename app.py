from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import io
import base64

app = Flask(_name_)
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Simple API Key (you can generate your own random one)
API_KEY = os.environ.get("API_KEY", "my-secret-key-123")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ Check for API key
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    # ✅ Check for image
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # ✅ Run YOLO prediction
    results = model.predict(image, save=True)
    result_image_path = results[0].save_dir + "/result.jpg"

    # ✅ Convert result image to base64
    with open(result_image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Optional cleanup
    try:
        os.remove(result_image_path)
    except:
        pass

    return jsonify({"result_image": img_b64})

if _name_ == "_main_":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)