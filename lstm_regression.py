# import liblaries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle 

from sklearn.model_selection import train_test_split # divide data to train and test  #veriyi eğitim ve test olmak uzere 2ye ayırır
from sklearn.preprocessing import minmax_scale #normalization

from tensorflow.keras.preprocessing.text import Tokenizer #tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences#padding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError


#load yelp dataset from huggingface https://huggingface.co/datasets/Yelp/yelp_review_full




#data preprocessing




#LSTM based regression model





#model compile and training 





# visulize training  loss graphs and save model 