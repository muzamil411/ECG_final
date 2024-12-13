from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) to allow React Native to access the API
CORS(app)

# Load the trained model (make sure to update the path if necessary)
model = load_model('ecg_classification_model.h5')

# Define class labels
class_labels = ['Myocardial Infarction', 'History of MI', 'Abnormal Heartbeat', 'Normal ECG']

# Define a function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Rescale the image
    return np.expand_dims(img_array, axis=0)

# Define route for homepage
@app.route('/')
def home():
    return 'Flask API is Running'

# Define route for uploading and predicting
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Preprocess the image
        processed_image = preprocess_image(file_path)

        # Predict the class
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_probabilities = predictions[0].tolist()

        # Remove the uploaded file after prediction
        os.remove(file_path)

        # Return the result
        return jsonify({
            'predicted_class': predicted_class_label,
            'probabilities': {label: prob for label, prob in zip(class_labels, predicted_probabilities)}
        })

# Run the app
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create the uploads folder if it doesn't exist
    app.run(debug=True)
