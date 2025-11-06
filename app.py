from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os
from PIL import Image
import base64

app = Flask(__name__)

# ✅ Enable CORS globally with support for headers and credentials
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
)

# ✅ Handle preflight (OPTIONS) requests globally
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 200

# ✅ Add headers to every response
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return response

# ✅ Load YOLO model once on startup
model = YOLO("best.pt")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ Validate file upload
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = Image.open(file.stream).convert("RGB")

    # ✅ Run YOLO detection
    results = model.predict(image, save=True)
    result_image_path = results[0].save_dir + "/result.jpg"

    # ✅ Encode result image as base64
    with open(result_image_path, "rb") as img_file:
        img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify({"result_image": img_b64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
