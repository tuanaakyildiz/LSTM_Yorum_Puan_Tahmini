# import liblaries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle 

from sklearn.model_selection import train_test_split # divide data to train and test  
from sklearn.preprocessing import MinMaxScaler #normalization

from tensorflow.keras.preprocessing.text import Tokenizer #tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences#padding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


#load yelp dataset from huggingface https://huggingface.co/datasets/Yelp/yelp_review_full
#huggingfaceden yelp veri setini yükle

splits = {"train":"yelp_review_full/train-00000-of-00001.parquet"}
train_path= "hf://datasets/Yelp/yelp_review_full/" + splits["train"]

#read data from parquet format
df= pd.read_parquet(train_path)
print(df.head())

#translate labels from 0-4 to 1-5
df["label"] = df["label"] + 1

#data preprocessing

texts = df["text"].values #review texts
labels = df["label"].values # scores are between 1-5

#tokenizer: translate text to numbers
#num_words most used first 10000
#OOV show unknown words with this label
tokenizer = Tokenizer(num_words = 10000, oov_token="<OOV>")

#texts to numbers
tokenizer.fit_on_texts(texts)

#save tokenizer
with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer, f)

# set review as a series
sequences = tokenizer.texts_to_sequences(texts)

#set all series to same length ad padding as zeros
padded_sequences = pad_sequences(sequences, maxlen =100, padding= "post", truncating="post")

#labels are between 1-5, 'Cause regression works more stabil between 0-1, lets set labels to 0-1 with normalization
scaler= MinMaxScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1,1))

#divide training and test datas
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,labels_scaled,test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train: {X_train[:2]}")

print(f"y_train shape: {y_train.shape}")
print(f"y_train: {y_train[:2]}")

#LSTM based regression model

model=Sequential()

#embedding layer: word indexs to vektor space
#input_dim: 10000 word count
#output_dim: 128 every word is set 128 dimension vektor
#input_lenght: constant series legth, length of every series
model.add(Embedding(input_dim = 10000, output_dim = 128, input_length=100))

#LSTM layer: train in sequential datas layer
model.add(LSTM(128)) # 128:lstm cell count learning capacity

# dense layer
model.add(Dense(64, activation = "relu"))

#output layer
model.add(Dense(1, activation="linear"))

#model compile and training 

model.compile(
    optimizer = "adam", # adaptive learning algorithm
    loss = MeanSquaredError(), # loss function suitable for regression
    metrics= [MeanAbsoluteError()] # models mean error
)

history = model.fit(
    X_train, y_train,
    epochs = 3,  # total training cycles
    batch_size = 64, # number of examples to be processed in each step.
    validation_split = 0.2 # 20% of the training data is set aside for validation.
)

# visulize training  loss graphs

plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Eğitim Süreci Loss: MSE")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

#save model

model.save("regression_lstm_yelp.h5")
