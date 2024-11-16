import numpy as np
import matplotlib.pyplot as plt
from model_lstm import build_lstm_model
from custom_callbacks import ThresholdCallback
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from datetime import datetime
import argparse

def train_lstm(X_train, X_val, y_train, y_val, config):
    """
    Train the LSTM model with the given data and configuration.
    
    Args:
        X_train (np.array): Training input data
        X_val (np.array): Validation input data
        y_train (np.array): Training target data
        y_val (np.array): Validation target data
        config (dict): Configuration parameters
    
    Returns:
        keras.Model: Trained LSTM model
    """
    logging.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")

    try:
        # Build model
        lstm_model = build_lstm_model((X_train.shape[1], 1), config['lstm_layers'], config['dropout_rate'])

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=config['reduce_lr_patience'], min_lr=1e-6)
        threshold_callback = ThresholdCallback(threshold=config['threshold'])

        # Train model
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stopping, reduce_lr, threshold_callback],
            verbose=1
        )

        # Plot Losses
        plt.figure(figsize=(12, 6))
        plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
        plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
        plt.title('Model Loss Comparison')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # Save plot
        plot_filename = f'lstm_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename)
        logging.info(f"Loss plot saved as {plot_filename}")
        plt.close()

        return lstm_model

    except Exception as e:
        logging.error(f"Error in training LSTM model: {e}")
        raise

def lstm_predict(X_test, lstm_model):
    """
    Make predictions using the trained LSTM model.
    
    Args:
        X_test (np.array): Test input data
        lstm_model (keras.Model): Trained LSTM model
    
    Returns:
        np.array: Predictions
    """
    try:
        lstm_preds = lstm_model.predict(X_test)
        return lstm_preds
    except Exception as e:
        logging.error(f"Error in LSTM prediction: {e}")
        raise

def get_config():
    """
    Get configuration from command line arguments.
    
    Returns:
        dict: Configuration parameters
    """
    parser = argparse.ArgumentParser(description='LSTM Model Training Configuration')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lstm_layers', type=int, nargs='+', default=[128, 64, 32], help='LSTM layer sizes')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--reduce_lr_patience', type=int, default=5, help='ReduceLROnPlateau patience')
    parser.add_argument('--threshold', type=float, default=1.0, help='Threshold for callback')

    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    config = get_config()
    logging.info(f"Configuration: {config}")
    