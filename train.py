import pandas as pd
import numpy as np
import requests
import os
import tensorflow as tf
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import json

data_path = 'wikiart_scraped.csv'
df = pd.read_csv(data_path)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_visual_art(image_paths):
    processed_art = []
    for image_path in image_paths:
        image = preprocess_image(image_path)
        features = base_model.predict(image)
        features = features / np.max(features)
        processed_art.append(features)
    processed_art = np.array(processed_art)
    return processed_art

def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False

image_dir = 'images'
os.makedirs(image_dir, exist_ok=True)
image_paths = []
for index, row in df.iterrows():
    style = row.get('Style')
    artwork = row.get('Artwork')
    artist = row.get('Artist')
    date = row.get('Date')
    link = row.get('Link')
    filename = f"{index}.jpg"
    save_path = os.path.join(image_dir, filename)
    success = download_image(link, save_path)
    if success:
        image_paths.append(save_path)
    if (index + 1) % 1000 == 0:
        print(f"Processed {index + 1} images.")

instructions = df['Artwork'].values
processed_art = preprocess_visual_art(image_paths)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(instructions)
tokenized_instructions = tokenizer.texts_to_sequences(instructions)
max_sequence_length = max(len(seq) for seq in tokenized_instructions)
padded_instructions = pad_sequences(tokenized_instructions, maxlen=max_sequence_length, padding='post')

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as json_file:
    json_file.write(tokenizer_json)

x_train, x_val, y_train, y_val = train_test_split(padded_instructions, processed_art, test_size=0.2, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

image_features = []
for image_path in image_paths:
    image = preprocess_image(image_path)
    features = base_model.predict(image)
    image_features.append(features)
image_features = np.array(image_features)

text_input = layers.Input(shape=(max_sequence_length,))
text_embedding = layers.Embedding(input_dim=max(tokenizer.word_index.values()) + 1, output_dim=128, input_length=max_sequence_length)(text_input)
text_conv = layers.Conv1D(128, 5, activation='relu')(text_embedding)
text_pooling = layers.GlobalMaxPooling1D()(text_conv)

image_input = layers.Input(shape=(7, 7, 512))
image_flattened = layers.Flatten()(image_input)
image_dense = layers.Dense(128, activation='relu')(image_flattened)

concatenated = layers.concatenate([text_pooling, image_dense])

interpretation = layers.Dense(128, activation='relu')(concatenated)
translation = layers.Dense(processed_art.shape[1], activation='softmax')(interpretation)

model = Model(inputs=[text_input, image_input], outputs=translation)
model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 32
epochs = 10

data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images = data_augmentation.flow(image_features, y_train, batch_size=batch_size)

model.fit([x_train, augmented_images], y_train, batch_size=batch_size, epochs=epochs, validation_data=([x_val, image_features], y_val))

model.save('art_translation_model.h5')

print(f"Total images processed: {len(df)}")
