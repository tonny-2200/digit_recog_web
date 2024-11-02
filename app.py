from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow import keras
import base64
import re

app = Flask(__name__)

# Load your trained model
model = keras.models.load_model("CNN.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the JSON request
    data = request.json['imageData']  # Use request.json to access JSON data
    
    # Extract the base64 image string
    image_data = re.sub('^data:image/png;base64,', '', data)
    image_data = base64.b64decode(image_data)

    # Convert the binary data to a numpy array
    img_array = np.frombuffer(image_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image: resize, normalize, and reshape
    img = cv2.resize(img, (28, 28))
    
    img = img.reshape(-1, 28, 28, 1) 
    img = img / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return jsonify({'prediction': str(predicted_digit)})  # Use jsonify to return JSON response

if __name__ == '__main__':
    app.run(debug=True)
