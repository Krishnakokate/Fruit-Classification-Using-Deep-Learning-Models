from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load models globally
models = {
    'InceptionV3': load_model('models/InceptionV3.h5'),
    'VGG19': load_model('models/VGG19.h5'),
    'ResNet50': load_model('models/ResNet50.h5')
}

# Define your custom class labels
class_labels = ['apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit', 'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit']  # Example labels

def predict_image(file_storage, model_name):
    model = models[model_name]

    img = Image.open(file_storage)
    img = img.resize((224, 224))  # Standardize to 224x224 for simplicity

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image according to the model
    if model_name == 'InceptionV3':
        processed_img = preprocess_input_inception(img_array)
    elif model_name == 'VGG19':
        processed_img = preprocess_input_vgg19(img_array)
    elif model_name == 'ResNet50':
        processed_img = preprocess_input_resnet50(img_array)

    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_index]

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/fruit-recognition', methods=['POST'])
def fruit_recognition():
    if request.method == 'POST':
        file = request.files['file']
        results = {}
        for model in models.keys():
            predicted_class = predict_image(file, model)
            results[model] = predicted_class
        return jsonify(results)  # Return JSON response

if __name__ == '__main__':
    app.run(debug=True)
