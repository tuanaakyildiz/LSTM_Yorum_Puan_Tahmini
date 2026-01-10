# import liblaries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle 

from sklearn.model_selection import train_test_split # divide data to train and test  #veriyi eğitim ve test olmak uzere 2ye ayırır
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

#parquet formatindan veriyi pandas ile oku
df= pd.read_parquet(train_path)
print(df.head())

#etiketleri 0-4 aralığından 1-5 aralığına dönüştürelim
df["label"] = df["label"] + 1

#data preprocessing

texts = df["text"].values #yorum metinlerimiz
labels = df["label"].values # puanlar 1-5 arasında

#tokenizer: metni sayiya cevir
#num_words en cok gecen ilk 10000
#OOV bilinmeyen kelimeleri bu etiketle göster
tokenizer = Tokenizer(num_words = 10000, oov_token="<OOV>")

#metni sayilara dönüştür
tokenizer.fit_on_texts(texts)

#tokenizer diske kaydet
with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer, f)

# yorumları dizi haline getir
sequences = tokenizer.texts_to_sequences(texts)

#tum dizileri sabit uzunluğa getir padding kisa olanları 0 ile doldur
padded_sequences = pad_sequences(sequences, maxlen =100, padding= "post", truncating="post")

#etiketler 1 ile 5 arasında, normalization ile 0 ile 1 arasına alalım cünkü regresyon problemlerinde daha stabil bir öğrenme sağlıyor
scaler= MinMaxScaler()
labels_scaled = scaler.fit_transform(labels.reshape(-1,1))

#eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,labels_scaled,text_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_train: {X_train[:2]}")

print(f"y_train shape: {y_train.shape}")
print(f"y_train: {y_train[:2]}")




#LSTM based regression model





#model compile and training 





# visulize training  loss graphs and save model 