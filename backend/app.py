from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import boto3
import onnxruntime as ort
import numpy as np
from dotenv import load_dotenv
from pymep.realParser import parse

import os
import base64
import io
from datetime import datetime
import time
import asyncio

from utils.image_processing import transform_image, process_image

load_dotenv()

app = Flask(__name__)
CORS(app, origins='*', methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

if os.getenv('ACCESS_KEY'):
    print("Found environment variable credentials. Using them to create session.")
    session = boto3.Session(
        aws_access_key_id=os.getenv('ACCESS_KEY'),
        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('SESSION_TOKEN')
    )
else:
    print("No environment variable credentials found. Relying on IAM role.")
    session = boto3.Session()

bucket = "penman-lln"
s3 = session.resource('s3')

# Loads the model once per instance of the server.
model = s3.Object(bucket, "model/penman_cnn.onnx").get()['Body'].read()

label_decoding = {
    10: '(',
    11: ')',
    12: '+',
    13: '-',
    14: '=',
    15: 'fwd_slash',
    16: 'times'
}

def upload(buffer, key):
    s3.Object(bucket, key).put(
        Body=buffer,
        ContentType='image/png'
    )

@app.route('/api/predict', methods=['POST'])
async def predict():
    response = request.get_json()
    image_data = response["imageData"].split(",", 1)[1]
    
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    # Convert to PIL image in order to compress it down to 28x28.
    image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_data, "utf-8"))))

    # Places equation in equation S3 bucket for continuous finetuning.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    key = f"data/equation/equation_{current_time}.png"

    # Uploads to bucket asynchronously to decrease latency.
    # asyncio.create_task(asyncio.to_thread(upload(buffer, key)))

    # Performs image transformations by cutting up, stretching, and morphing data.
    tokens = process_image(image)

    # Creates a session with the model and feeds the expression through the model.
    session = ort.InferenceSession(model)
    output = session.run(None, {"X": tokens.astype(np.float32)})[0]

    predicted_values = np.argmax(output, axis=1).tolist()
    confidence = np.max(output, axis=1).tolist()

    # Converts to string representation.
    predicted_values = [label_decoding[val] if val >= 10 else str(val) for val in predicted_values]

    expression = "".join(predicted_values)

    try:
        # TODO: Maybe consider implementing floats in the future.
        value = str(int(parse(expression)))
    except Exception as e:
        return jsonify(
            {
                "error": f"Failed to parse expression. Got {expression}, expected actual expression.\n{e}"
            }
        ), 400

    return jsonify(
        {
            'predicted_values': predicted_values,
            'confidence': confidence,
            'determined_value': value
        }
    ), 200

@app.route('/api/save_image', methods=["POST"])
async def save_image():
    start_time = time.time()

    response = request.get_json()
    image_data = response["imageData"].split(",", 1)[1]
    
    current_time = datetime.now().strftime("%H-%M-%S")

    # Convert to PIL image in order to compress it down to 28x28.
    image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_data, "utf-8"))))

    # Places equation in equation S3 bucket asynchronously for continued finetuning.
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    key = f"data/token/=/token_{current_time}.png"

    curr_time = time.time()
    print(f"Time to process image: {curr_time - start_time}")

    asyncio.create_task(asyncio.to_thread(upload(buffer, key)))

    print(f"Time to async: {time.time() - curr_time}")

    return jsonify(), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
