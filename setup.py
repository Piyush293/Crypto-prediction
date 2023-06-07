import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf

import os
import re
import io
import nltk
import json
from toml import load

from keras.models import load_model
import streamlit as st

import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.backend as K
import seaborn as sns

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Bidirectional, SpatialDropout1D
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,LSTM
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import History
from tensorflow.keras import losses
from tensorflow.keras.preprocessing import sequence


def crypto(user_input, start, end):
    
    data = yf.download(user_input,start,end)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    preddays = 30

    # Splitting data
    model = load_model('keras_model.h5')

    #Testing Part

    test_start = dt.datetime(2021, 10, 1)
    test_end = dt.datetime.now()
    test_data = data.loc[(data.index >= test_start) & (data.index <= test_end)]
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - preddays:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(preddays,  len(model_inputs)):
        x_test.append(model_inputs[x-preddays:x, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred = model.predict(x_test)
    pred = scaler.inverse_transform(pred)

    return("The Predicted price of Bitcoin for today is INR: " + str(float(pred[-1])))



