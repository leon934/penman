from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import boto3

import os
import base64
import io
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

session = boto3.Session(
    aws_access_key_id=os.getenv('ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
    aws_session_token=os.getenv('SESSION_TOKEN')
)

s3 = session.resource('s3')

@app.route('/image', methods=['POST'])
def save_image():
    response = request.get_json()
    image_data = response["imageData"].split(",", 1)[1]
    
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")

    s3_obj = s3.Object("penman-lln", f"data/equation/equation_{current_time}.png")
    s3_obj.put(
        Body=image_data,
        ContentType='image/png'
    )

    # file = request.files['file']
    # filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(filepath)

    # resizeImage()

    return jsonify({'message': 'Snapshot saved successfully.'}), 200

if __name__ == "__main__":
    app.run(debug=True)