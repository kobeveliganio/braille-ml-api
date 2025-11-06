from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ import CORS
from ultralytics import YOLO
import os
from PIL import Image
import base64

# ❌ You had "name" instead of "__name__"
app = Flask(__name__)

# ✅ Enable CORS for all routes and origins (for local dev)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Load your YOLO model
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

    # ✅ Run YOLO detection
    results = model.predict(image, save=True)

    # ✅ Get result path correctly
    result_image_path = os.path.join(results[0].save_dir, "image0.jpg")
    if not os.path.exists(result_image_path):
        return jsonify({"error": "Result image not found"}), 500

    with open(result_image_path, "rb") as img_file:
        img_bytes = img_file.read()

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return jsonify({"result_image": img_b64})

# ❌ You had "if name == 'main':"
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses $PORT env var
    app.run(host="0.0.0.0", port=port)
