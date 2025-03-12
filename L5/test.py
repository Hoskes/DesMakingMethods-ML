import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Параметры
SEQ_LENGTH = 100
BATCH_SIZE = 256
EMBEDDING_DIM = 64
RNN_UNITS = 128
EPOCHS = 10

# 1. Загрузка и подготовка данных
df = pd.read_csv('lenta-ru.csv', usecols=['text'], encoding='utf-8')
texts = df['text'].str.lower().dropna().str.replace('[^\w\s]', '', regex=True).tolist()
print(texts)