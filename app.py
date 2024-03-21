from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load your trained model
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if necessary
    img = cv2.equalizeHist(img)  # Apply histogram equalization
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Adjust the input shape of the model
input_shape = (256, 256, 1)  # Assuming grayscale images of size 256x256

# Load your trained model
model = joblib.load('/Users/faizanhabib/Desktop/Vitamin-D kit project/svr_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML template for the form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image file
    image_path = 'temp_image.jpg'  # Temporarily save the image file
    image_file.save(image_path)

    # Preprocess the image
    target_size = (256, 256)
    preprocessed_image = preprocess_image(image_path, target_size)

    # Make prediction
    prediction = model.predict(np.expand_dims(preprocessed_image.flatten(), axis=0))[0]

    # Delete the temp image file
    os.remove(image_path)

    # Prepare response
    response = {
        "success": True,
        "message": "Prediction successful.",
        "payload": {
            "prediction": float(prediction)
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
