from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load DeepLab v3 from TF Hub (ADE20K-trained)
MODEL_URL = "https://tfhub.dev/tensorflow/deeplabv3/1"  # ADE20K
model = hub.load(MODEL_URL)

# ADE20K label for "floor" (index may vary—ADE20K standard: 7 is 'floor')
FLOOR_CLASS = 7

def segment_floor(image: Image.Image):
    img = image.convert("RGB")
    # Resize to model input size (e.g. 513)
    target_size = 513
    img_resized = img.resize((target_size, target_size))
    inp = tf.expand_dims(np.array(img_resized) / 255.0, 0)
    result = model(inp)
    seg_map = tf.argmax(result['default'], axis=3)[0].numpy().astype(np.uint8)
    # Resize seg_map back to original
    seg_map = Image.fromarray(seg_map).resize(img.size, resample=Image.NEAREST)
    seg_arr = np.array(seg_map)
    # Create a binary mask image where floor pixels are white, others transparent
    mask = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
    mask[seg_arr == FLOOR_CLASS, :] = [255,255,255,255]
    return Image.fromarray(mask, mode="RGBA")

@app.route("/segment", methods=["POST"])
def segment_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "no image"}), 400
    file = request.files['image']
    img = Image.open(file.stream)
    mask = segment_floor(img)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
