from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    image_file = request.files.get('file')
    if image_file:
        # Preprocess the image as per your model's input requirements
        image = load_img(image_file, target_size=(224, 224))  # Use the correct size
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Model expects a batch dimension

        # Make a prediction
        prediction = model.predict(image)

        # Process the prediction result (e.g., class labels)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify({'predicted_class': int(predicted_class)})
    else:
        return jsonify({'error': 'No file provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
