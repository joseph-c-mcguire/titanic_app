# app.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.joblib')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)