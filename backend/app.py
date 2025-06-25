from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import boto3
import onnxruntime as ort
import numpy as np
from dotenv import load_dotenv

import os
import base64
import io
from datetime import datetime
import json

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

def process_image(image: Image):
    # Resizes and inverts image so the larger dimension is 28 and is the proper format for the CNN.
    image = resize_image(image)

@app.route('/image', methods=['POST'])
def save_image():
    response = request.get_json()
    image_data = response["imageData"].split(",", 1)[1]
    
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    # Places equation in equation S3 bucket for further usage (?)
    s3_obj = s3.Object("penman-lln", f"data/equation/equation_{current_time}.png")
    s3_obj.put(
        Body=image_data,
        ContentType='image/png'
    )

    # Convert to PIL image in order to compress it down to 28x28.
    image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_data, "utf-8"))))

    print(image_data)

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

    # Obtains the tldraw JSON representation of the number from the S3 bucket.
    tldraw_nums = []
    
    for val in predicted_values:
        print(val)

        content = s3.Object("penman-lln", f"data/evaluation_digits/{val}.json").get()['Body'].read().decode("utf-8")
        tldraw_nums.append(json.loads(content))

    return jsonify(
        {
            'predicted_values': predicted_values,
            'confidence': confidence,
            'tldraw_numbers': tldraw_nums
        }
    ), 200

if __name__ == "__main__":
    app.run(debug=True)