from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from flask!"})

@app.route('/image', methods=['POST'])
def save_image():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return jsonify({'message': 'Snapshot saved successfully.'}), 200

if __name__ == "__main__":
    app.run(debug=True)