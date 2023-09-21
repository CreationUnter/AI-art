from flask import Flask, request, render_template
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from PIL import Image
import pandas as pd
from keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

model = load_model('art_translation_model.h5')

with open('tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())

def preprocess_text(text):
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
    return padded_text

def preprocess_image(image):
    processed_image = image.resize((224, 224))
    processed_image = np.array(processed_image)
    processed_image = preprocess_input(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

data_path = 'wikiart_scraped.csv'
df = pd.read_csv(data_path)

instructions = df['Artwork'].values
tokenized_instructions = tokenizer.texts_to_sequences(instructions)
max_sequence_length = max(len(seq) for seq in tokenized_instructions)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    image_file = request.files['image']
    processed_text = preprocess_text(text)
    processed_image = preprocess_image(image_file)
    prediction = model.predict([processed_text, processed_image])
    predicted_label = np.argmax(prediction)
    label_to_class = {0: 'Abstract', 1: 'Realism', 2: 'Impressionism'}
    predicted_class = label_to_class[predicted_label]
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
