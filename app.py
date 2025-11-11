from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import cv2
import time

app = Flask(__name__)

# ✅ Allow requests from your frontend domain only
CORS(app, origins=["https://smartvision-betl.onrender.com"])  

# ✅ Folders for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ✅ Load YOLO model
try:
    model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print("❌ Failed to load YOLO model:", e)
    model = None

# ✅ Serve annotated images
@app.route('/results_images/<filename>')
def serve_results_image(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Braille YOLO API is running."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "YOLO model not loaded"}), 500

        file = request.files.get('file') or request.files.get('image')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save uploaded file
        timestamp = int(time.time())
        file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{file.filename}")
        file.save(file_path)

        # Run YOLO prediction
        results = model.predict(source=file_path, conf=0.25, verbose=False)

        # Annotate image
        annotated_img = results[0].plot()

        # Save annotated image
        annotated_filename = f"annotated_{timestamp}_{file.filename}"
        annotated_path = os.path.join(RESULTS_FOLDER, annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)

        # Build predictions array
        predictions = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            predictions.append({"label": label})

        translated_text = "".join([p["label"][0].upper() for p in predictions])

        result = {
            "annotated_image_path": f"results_images/{annotated_filename}",
            "braille_text": [p["label"] for p in predictions],
            "translated_text": translated_text
        }

        return jsonify(result)

    except Exception as e:
        print("🔥 Server error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
