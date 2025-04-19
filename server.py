from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained pneumonia detection model when the application starts.
model = load_model('pneumonia_model.h5')

def predict_image(image_path):
    """
    Function to load an image and prepare it for prediction.
    The image is converted to grayscale (single channel) and resized to 160×160 pixels.

    :param image_path: The path of the patient's image.
    :return: The predicted label ('PNEUMONIA' or 'NORMAL') and the confidence score.
    """
    # Load the image in grayscale mode and resize it to 160×160 pixels.
    img = image.load_img(image_path, target_size=(160, 160), color_mode="grayscale")

    # Convert the image to a numerical array.
    img_array = image.img_to_array(img)

    # Normalize the array values to the range [0, 1].
    img_array = img_array / 255.0

    # Expand the dimensions to add a batch dimension.
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to predict the image.
    pred_value = model.predict(img_array)[0][0]

    # Determine the label based on a threshold value of 0.5.
    # If pred_value is less than or equal to 0.5, label as "PNEUMONIA" and confidence is 1 - pred_value.
    # Otherwise, label as "NORMAL" and confidence is equal to pred_value.
    if pred_value <= 0.5:
        label = "PNEUMONIA"
        confidence = 1 - pred_value
    else:
        label = "NORMAL"
        confidence = pred_value

    return label, confidence


@app.route('/')
def index():
    # Render the index.html page.
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file was provided in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded!'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected!'}), 400

    # Save the file temporarily in the "uploads" folder.
    upload_dir = os.path.join(os.getcwd(), "uploads")
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)

    # Predict using the saved image.
    label, confidence = predict_image(file_path)

    # Optionally remove the uploaded file after processing.
    os.remove(file_path)

    # Return the prediction and confidence as a JSON response.
    return jsonify({
        'prediction': label,
        'confidence': float(confidence)
    })


if __name__ == '__main__':
    # Start the Flask application in debug mode.
    app.run(debug=True)
