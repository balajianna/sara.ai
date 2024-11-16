# Description: This script is used to train and save the Keras model for the project milestone
# Author: Balaji Anna
# Last Edited: 2024-Nov-16

import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from data_preprocessing import process_subjects
from train_lstm_backup import get_config, train_lstm
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
import configparser
# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Get parameters from config file
DIRECTORY = config['DATA']['directory']
TIME_WINDOW = config.getint('DATA', 'time_window')
NUM_SUBJECTS = config.getint('DATA', 'num_subjects')
VALIDATION_SPLIT = config.getfloat('DATA', 'validation_split')
RANDOM_STATE = config.getint('DATA', 'random_state')

EPOCHS = config.getint('TRAINING', 'epochs')
BATCH_SIZE = config.getint('TRAINING', 'batch_size')
THRESHOLD = config.getfloat('TRAINING', 'threshold')

KERAS_MODEL_PATH = config['MODEL']['keras_model_path']
SCALAR_PATH = config['MODEL']['scaler_path']

LOG_PATH = config['LOG']['log_path']

# Set up logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_and_save():
    try:
        # Step 1: Data Preprocessing
        logging.info(f"Preprocessing data for {NUM_SUBJECTS} subjects...")
        X, y, scaler = process_subjects(DIRECTORY, NUM_SUBJECTS, TIME_WINDOW)
        if X is None or y is None:
            raise ValueError("Data processing failed.")

        # Step 2: Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)
        logging.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}")

        # Get configuration
        config = get_config()
        logging.info(f"Configuration: {config}")

        # Step 3: Train LSTM model
        logging.info("Training LSTM model...")
        lstm_model, history = train_lstm(
            X_train[:, :, np.newaxis],
            X_val[:, :, np.newaxis],
            y_train,
            y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            threshold=THRESHOLD
        )

        # Plot and save the loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        loss_plot_filename = f'lstm_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(loss_plot_filename)
        plt.close()
        logging.info(f"Loss plot saved as {loss_plot_filename}")

        # Step 4: Save the model and scaler
        model_filename = KERAS_MODEL_PATH
        lstm_model.save(model_filename)
        logging.info(f"Model saved as {model_filename}")

        scaler_filename = SCALAR_PATH
        joblib.dump(scaler, scaler_filename)
        logging.info(f"Scaler saved as {scaler_filename}")

        return model_filename, scaler_filename
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        return None, None

if __name__ == "__main__":
    model_file, scaler_file = train_and_save()
    if model_file and scaler_file:
        print(f"Training complete. Model: {model_file}, Scaler: {scaler_file}")
    else:
        print("Training failed. Check the log for details.")