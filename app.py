from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app) # to connect to React

#Load Model
model = load_model("regression_lstm_yelp.h5", compile=False)

#Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

#Rest API Endpoint

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocessing
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding="post")
    
    # Prediction
    prediction = model.predict(padded)
    score = float(prediction[0][0] * 5) # 0-5 skalasına çek
    
    return jsonify({'score': round(score, 2)})

#to strat Flask API

if __name__ == '__main__':
    app.run(debug=True, port=5000)