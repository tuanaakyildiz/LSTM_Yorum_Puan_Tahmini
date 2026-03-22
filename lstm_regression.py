# ==========================================
# LSTM SENTIMENT ANALYSIS: TRAINING SCRIPT
# ==========================================

# --- 1. LIBRARY IMPORTS ---
print(">>> [1/8] Importing required libraries...")
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle 
import sys

from sklearn.model_selection import train_test_split # To divide data into train and test sets
from sklearn.preprocessing import MinMaxScaler # For mathematical normalization of scores

from tensorflow.keras.preprocessing.text import Tokenizer # Translates text into numbers
from tensorflow.keras.preprocessing.sequence import pad_sequences # Ensures sequences are the same length
from tensorflow.keras.models import Sequential
# Importing advanced layers to make the model highly sensitive to strong emotions
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GlobalMaxPooling1D, Dropout

# --- 2. DATA LOADING ---
print(">>> [2/8] Loading the Yelp dataset from Hugging Face...")
try:
    # Load Yelp dataset from Hugging Face parquet format
    splits = {"train": "yelp_review_full/train-00000-of-00001.parquet"}
    train_path = "hf://datasets/Yelp/yelp_review_full/" + splits["train"]
    
    df = pd.read_parquet(train_path)
    print(f"    -> Successfully loaded {len(df)} rows of data.")
except Exception as e:
    # Graceful exit if the download fails, rather than a massive traceback
    print(f"    -> ERROR loading data: {e}")
    sys.exit(1)

# Original Yelp labels are 0-4. We shift them to 1-5 for human readability.
df["label"] = df["label"] + 1

texts = df["text"].values # The actual review text
labels = df["label"].values # The 1-5 scores

# --- 3. TEXT PREPROCESSING & TOKENIZATION ---
print(">>> [3/8] Initializing Tokenizer and fitting on texts...")
# We limit the model to the 10,000 most frequently used words to optimize memory
MAX_WORDS = 10000
MAX_LEN = 100 # We will cut or pad every review to exactly 100 words

# Initialize Tokenizer. Unknown words will be tagged as <OOV> (Out Of Vocabulary)
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# FIX: Defining vocab_size properly. 
# We must add +1 because index '0' is technically reserved by Keras for padding zeros.
vocab_size = len(tokenizer.word_index) + 1 
print(f"    -> Total unique words found: {vocab_size - 1}")
print(f"    -> Vocabulary size set for Embedding layer: {vocab_size}")

print(">>> [4/8] Saving Tokenizer to tokenizer.pkl...")
# Save the tokenizer so the Flask API can understand user input later
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# --- 4. SEQUENCE PADDING ---
print(">>> [5/8] Converting texts to sequences and applying padding...")
# Translate the English words into their integer dictionary equivalents
sequences = tokenizer.texts_to_sequences(texts)

# Ensure every review is exactly 100 integers long. 
# Shorter reviews get zeros at the end ("post"), longer ones get cut at the end ("post").
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
print(f"    -> Padded sequences shape: {padded_sequences.shape}")

# --- 5. LABEL NORMALIZATION ---
print(">>> [6/8] Normalizing labels (1-5) to scale (0.0-1.0)...")
# Deep learning models struggle to predict arbitrary ranges like 1-5. 
# They work best when target values are mathematically squashed between 0.0 and 1.0.
scaler = MinMaxScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# Save the scaler so the backend can reverse the math (0.0-1.0 back to 1-5) later
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data: 80% for training the AI, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels_scaled, test_size=0.2, random_state=42)
print(f"    -> X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

# --- 6. ENHANCED LSTM ARCHITECTURE ---
print(">>> [7/8] Compiling the Enhanced Bidirectional LSTM model...")
# This architecture specifically solves the issue of the model always guessing ~3.0
model = Sequential([
    # 1. Embedding Layer: Maps word integers to a 128-dimensional continuous vector space
    Embedding(input_dim=vocab_size, output_dim=128, input_length=MAX_LEN),
    
    # 2. Bidirectional LSTM: Reads the sequence forwards AND backwards.
    # This captures complex context, like "Not (end of sentence) ... good (start of sentence)".
    Bidirectional(LSTM(64, return_sequences=True)),
    
    # 3. GlobalMaxPooling1D: The "Neutrality Fix". 
    # Instead of averaging the sentiment of the whole sentence (which results in 3.0),
    # this layer acts like a magnet, pulling out only the absolute strongest emotional signals.
    GlobalMaxPooling1D(), 
    
    # 4. Dense processing layer to interpret the pulled features
    Dense(32, activation='relu'),
    
    # 5. Dropout: Randomly turns off 40% of neurons during training to prevent memorization (overfitting)
    Dropout(0.4), 
    
    # 6. Output Layer: Sigmoid forces the final prediction to be a probability between 0.0 and 1.0
    Dense(1, activation='sigmoid') 
])

# Compile the model
model.compile(
    optimizer="adam", # Adaptive learning algorithm
    loss='mean_squared_error', # MSE is mathematically superior for regression (predicting scores)
    metrics=['mean_absolute_error']
)
print("    -> Model compiled successfully.")
model.summary()

# --- 7. TRAINING THE MODEL ---
print(">>> [8/8] Starting model training...")
print("    -> Watch the progress bar below for Epoch updates.")

# verbose=1 ensures the terminal shows the live progress bar for each epoch
history = model.fit(
    X_train, y_train,
    epochs=5, # 5 epochs is usually sufficient to avoid overfitting on large datasets
    batch_size=128, # Processing 128 reviews at a time speeds up training
    validation_split=0.1,
    verbose=1 
)

print(">>> [SUCCESS] Training completed! Saving the model...")
model.save("regression_lstm_yelp.h5") # Save the trained weights
print("    -> Model saved as 'regression_lstm_yelp.h5'.")

# --- 8. VISUALIZATION ---
print(">>> Generating training loss graph...")
# Plotting the loss helps us see if the model actually learned or just memorized
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.plot(history.history["val_loss"], label="Validation Loss (MSE)")
plt.title("Model Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()