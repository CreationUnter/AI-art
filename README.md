
# Art Translation Project

This project uses a deep learning model to predict the artistic style of an artwork based on its description and visual content. It's a Flask web application that allows users to input text descriptions and upload images of artworks, and then it predicts whether the style of the artwork is "Abstract," "Realism," or "Impressionism."

The application will process your input and display the predicted artistic style as "Abstract," "Realism," or "Impressionism."

## Data
The model was trained on a dataset of artwork descriptions and images. The dataset is stored in a CSV file named wikiart_scraped.csv. You can replace this dataset with your own data if desired.


## How to start

To deploy this project run

```bash
Prerequisites
Python 3.x

# Install the required Python packages using requirements.txt. You can install them using pip:
pip install -r requirements.txt

# Running the Web Application
Clone the repository to your local machine:
git clone https://github.com/yourusername/art-translation-project.git

Navigate to the project directory:
cd art-translation-project

Start the Flask application by running app.py:
python app.py

The application will run locally, and you can access it in your web browser at http://localhost:5000.
```

## Project Structure 

train.py: This script is used to train the deep learning model using a dataset of artwork descriptions and images. It saves the trained model and tokenizer.

app.py: This is the Flask web application that uses the trained model to make predictions. It provides a simple interface for users to input descriptions and images.

tokenizer.json: A JSON file containing the tokenizer used to preprocess text descriptions.

art_translation_model.h5: The trained deep learning model for style prediction.

images/: A directory where images of artworks are stored.

## Acknowledgements

 This project uses the Keras deep learning library and the Flask web framework. The deep learning model is based on a combination of text and image processing using pre-trained models.

Feel Free to modify and extend this project for your own use or as a starting point for other art related machine learning applications.
