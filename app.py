from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained object detection model here
# Replace 'model.h5' with the path to your model file
model = tf.keras.models.load_model('model\model.h5')

# Define class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to the input size of your model (32x32 pixels)
    image = image.resize((32, 32))
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

# Define a function to predict the image
def predict_image(image, model, class_names):
    # Preprocess the image
    img = preprocess_image(image)
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]  # Get the class name

    # Print the predicted class name
    print("Predicted class:", class_name)

    return class_name

# Define a route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        uploaded_image = request.files['image']
        
        # Ensure an image was uploaded
        if uploaded_image.filename != '':
            # Open and preprocess the image
            image = Image.open(uploaded_image)
            
            # Make a prediction using your model
            class_name = predict_image(image, model, class_names)

            # Save the input image temporarily
            temp_buffer = io.BytesIO()
            image.save(temp_buffer, format="JPEG")

            # Return the result as JSON along with the image URL
            return jsonify({
                'image_url': '/uploads/' + uploaded_image.filename,
                'class_name': class_name
            })

        else:
            return jsonify({'error': 'No file uploaded'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(f'./uploads/{filename}', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
