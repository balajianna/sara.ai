# Description: This file contains the LSTM model architecture
# Author: Balaji Anna
# Last Edited: 2024-Nov-17

from tensorflow import keras
from keras import models, layers, optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
import logging
from tensorflow.keras.initializers import GlorotUniform

import configparser
# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Get parameters from config file
LEARNING_RATE = config.getfloat('TRAINING', 'learning_rate')
OPTIMIZER = config['TRAINING']['optimizer']
LOSS = config['TRAINING']['loss']
METRICS = config['TRAINING']['metrics']
DENSE_UNIT = config.getint('TRAINING', 'dense_unit')
ACTIVATION = config['TRAINING']['activation']

def build_lstm_model(input_shape, lstm_layers, dropout_rate):
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))
    model.add(LSTM(lstm_layers[0], return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for units in lstm_layers[1:-1]:
        model.add(LSTM(units, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Last LSTM layer
    model.add(LSTM(lstm_layers[-1]))
    model.add(BatchNormalization())

    # Output layers
    model.add(Dense(DENSE_UNIT, activation=ACTIVATION))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=LEARNING_RATE,  beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss=LOSS, metrics=[METRICS])

    logging.info("LSTM model summary:")
    model.summary(print_fn=logging.info)

    return model