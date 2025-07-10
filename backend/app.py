from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import boto3
import onnxruntime as ort
import numpy as np
from dotenv import load_dotenv
import aioboto3

import os
import base64
import io
from datetime import datetime
import json
import time
import asyncio

from algorithm.image_processing import resize_image, transform_single_digit

load_dotenv()

app = Flask(__name__)
CORS(app)

# TODO: Remove this when the code is in the EC2 instance; creating a session isn't required (i think).
session = boto3.Session(
    aws_access_key_id=os.getenv('ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('SESSION_TOKEN')
)

bucket = "penman-lln"
s3 = session.resource('s3')

# Loads the model once per instance of the server.
model = s3.Object("penman-lln", "model/penman_cnn.onnx").get()['Body'].read()

label_decoding = {
    10: '(',
    11: ')',
    12: '+',
    13: '-',
    14: '=',
    15: 'fwd_slash',
    16: 'times'
}

# TODO: Create a function to split up the image into its individual 
def process_image(image: Image):
    # Resizes and inverts image so the larger dimension is 28 and is the proper format for the CNN.
    image = resize_image(image)

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

    # Places equation in equation S3 bucket for further usage (?)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    key = f"data/equation/equation_{current_time}.png"

    # Uploads to bucket asynchronously to decrease latency.
    asyncio.create_task(asyncio.to_thread(upload(buffer, key)))

    # print(image_data)

    # FIXME: This ONLY works with a singular digit. Eventually, this should be adapted to work for more complex expressions.
    # Matrix shape required to feed into model.
    expression = transform_single_digit(image).reshape(1, 1, 28, 28)

    # Creates a session with the model and feeds the expression through the model.
    session = ort.InferenceSession(model)
    output = session.run(None, {"X": expression.astype(np.float32)})[0]

    predicted_values = np.argmax(output, axis=1).tolist()
    confidence = np.max(output, axis=1).tolist()

    # Converts to string representation.
    predicted_values = [label_decoding[val] if val >= 10 else val for val in predicted_values]

    return jsonify(
        {
            'predicted_values': predicted_values,
            'confidence': confidence,
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

if __name__ == "__main__":
    app.run(debug=True)