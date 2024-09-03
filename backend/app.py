from flask import Flask, jsonify, request
from flask_cors import CORS
import os
# from backend.algorithm.algorithm.image_processing import transformImage, findSymbols, resizeImage

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/image', methods=['POST'])
def save_image():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # resizeImage()

    return jsonify({'message': 'Snapshot saved successfully.'}), 200

if __name__ == "__main__":
    app.run(debug=True)