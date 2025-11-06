from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ Handles CORS automatically
from ultralytics import YOLO
import os
from PIL import Image
import base64

app = Flask(__name__)

# ✅ Allow specific origins (your localhost + your deployed frontend if you have one)
CORS(app, origins=["http://localhost:5173", "https://your-frontend-domain.com"])

# Load YOLO model
model = YOLO("best.pt")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # ✅ Handle OPTIONS preflight request
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # Run YOLO detection
    results = model.predict(image, save=True)
    result_image_path = os.path.join(results[0].save_dir, "image0.jpg")

    if not os.path.exists(result_image_path):
        return jsonify({"error": "Result image not found"}), 500

    with open(result_image_path, "rb") as img_file:
        img_bytes = img_file.read()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # ✅ Add CORS headers in the response
    response = jsonify({"result_image": img_b64})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
