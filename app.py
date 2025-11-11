from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import time
from supabase import create_client, Client
import base64

app = Flask(__name__)
CORS(app)

# Initialize YOLO model
model = YOLO("best.pt")

# Supabase configuration (store in env variables)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # Use service key for server uploads
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file") or request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded file temporarily
    timestamp = int(time.time())
    temp_filename = f"temp_{timestamp}_{file.filename}"
    temp_path = os.path.join("/tmp", temp_filename)
    file.save(temp_path)

    # Run YOLO prediction
    results = model.predict(source=temp_path, conf=0.25, verbose=False)

    # Annotate image
    annotated_img = results[0].plot()  # numpy array
    annotated_filename = f"annotated_{timestamp}_{file.filename}"
    annotated_path = os.path.join("/tmp", annotated_filename)
    cv2.imwrite(annotated_path, annotated_img)

    # Upload annotated image to Supabase storage
    with open(annotated_path, "rb") as f:
        file_bytes = f.read()

    storage_path = f"uploaded_works/{annotated_filename}"
    response = supabase.storage.from_("ml-server").upload(storage_path, file_bytes, {"upsert": True})

    if response.get("error"):
        return jsonify({"error": f"Supabase upload failed: {response['error']['message']}" }), 500

    # Get public URL
    public_url = supabase.storage.from_("ml-server").get_public_url(storage_path).get("publicUrl")

    # Build predictions array
    predictions = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = results[0].names[cls_id]
        predictions.append({"label": label, "confidence": round(confidence, 2)})

    translated_text = "".join([p["label"][0].upper() for p in predictions])

    return jsonify({
        "annotated_image_url": public_url,
        "braille_text": [p["label"] for p in predictions],
        "translated_text": translated_text
    })
