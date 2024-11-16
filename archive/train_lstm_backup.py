# Description: This script is used to train the LSTM model for the project milestone
# Author: Balaji Anna
# Last Edited: 2024-Nov-16

import numpy as np
import matplotlib.pyplot as plt
from model_lstm import build_lstm_model
from custom_callbacks import ThresholdCallback
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime
import argparse
import configparser
import joblib  # For saving the scaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)
h5_model_path = config['MODEL']['h5_model_path']

def get_config():
    parser = argparse.ArgumentParser(description='LSTM Model Training Configuration')
    parser.add_argument('--epochs', type=int, default=config.getint('TRAINING', 'epochs'), help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.getint('TRAINING', 'batch_size'), help='Batch size')
    
    # Parse lstm_layers as a list of integers
    default_lstm_layers = eval(config.get('TRAINING', 'lstm_layers'))
    parser.add_argument('--lstm_layers', type=int, nargs='+', default=default_lstm_layers, help='LSTM layer sizes')
    
    parser.add_argument('--dropout_rate', type=float, default=config.getfloat('TRAINING', 'dropout_rate'), help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=config.getint('TRAINING', 'early_stopping_patience'), help='Early stopping patience')
    parser.add_argument('--reduce_lr_patience', type=int, default=config.getint('TRAINING', 'reduce_lr_patience'), help='ReduceLROnPlateau patience')
    parser.add_argument('--threshold', type=float, default=config.getfloat('TRAINING', 'threshold'), help='Threshold for callback')
    args = parser.parse_args()
    return vars(args)

def train_lstm(X_train, X_val, y_train, y_val, epochs, batch_size, threshold):
    """
    Train the LSTM model with the given data and configuration.
    """
    logging.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")

    try:
        config = get_config()

        # Build model
        lstm_model = build_lstm_model(
            (X_train.shape[1], 1),
            config['lstm_layers'],
            config['dropout_rate']
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=config['reduce_lr_patience'],
            min_lr=1e-6
        )

        threshold_callback = ThresholdCallback(threshold=threshold)

        history = lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, threshold_callback]
        )

        y_pred = lstm_model.predict(X_val)
        mse, mae, rmse = calculate_metrics(y_val, y_pred)
        logging.info(f"Validation Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        # Plot Losses
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='LSTM Training Loss')
        plt.plot(history.history['val_loss'], label='LSTM Validation Loss')
        plt.title('Model Loss Comparison')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plot_filename = f'lstm_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename)
        logging.info(f"Loss plot saved as {plot_filename}")
        plt.close()

        # Plot learning rate
        if 'lr' in history.history:
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate over Epochs')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.legend()
            lr_plot_filename = f'learning_rate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(lr_plot_filename)
            logging.info(f"Learning rate plot saved as {lr_plot_filename}")
            plt.close()
        else:
            logging.warning("Learning rate history not available in the model history.")

        return lstm_model, history

    except Exception as e:
        logging.error(f"Error in training LSTM model: {str(e)}")
        raise

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

if __name__ == "__main__":
    config = get_config()
    logging.info(f"Configuration: {config}")

