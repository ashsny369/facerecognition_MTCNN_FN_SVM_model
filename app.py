
import os
import cv2
from flask_cors import CORS
import numpy as np
import base64
from flask import Flask, request, jsonify
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pickle
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Initialize FaceNet and MTCNN
embedder = FaceNet()
detector = MTCNN()
#encoder = LabelEncoder()


# Define the path to the SVM model file
svm_model_file = 'svm_model.pkl'

# Check if the SVM model file exists
if not os.path.exists(svm_model_file):
    raise FileNotFoundError(f"SVM model file '{svm_model_file}' not found")

with open(svm_model_file, 'rb') as f:
    svm_model = pickle.load(f)

# In-memory storage for image chunks
from collections import defaultdict
image_chunks = defaultdict(list)

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Endpoint for face recognition
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email = data.get('email')
    chunk = data.get('chunk')
    sequence_number = data.get('sequenceNumber')
    is_last_chunk = data.get('isLastChunk')

    if email not in image_chunks:
        image_chunks[email] = []

    image_chunks[email].append(chunk)

    if not is_last_chunk:
        return jsonify({"key": 1}), 200

    # Combine chunks and process the complete image
    image_base64 = ''.join(image_chunks.pop(email))
    image_data = base64.b64decode(image_base64)
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Detect face
    results = detector.detect_faces(image)
    if not results:
        return jsonify({'error': 'No face detected', "key":0}), 400
    
    # Process each detected face
    predictions = []
    for result in results:
        x1, y1, w, h = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        
        # Get embedding
        embedding = get_embedding(face)
        
        # Predict label
        prediction = svm_model.predict([embedding])
        #label = encoder.inverse_transform(prediction)[0]
        
        predictions.append(prediction)
    
    print(predictions)
    return jsonify({ "key": 2}), 200

if __name__ == '__main__':
    app.run(debug=True)
