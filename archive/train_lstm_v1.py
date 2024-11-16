import numpy as np
import matplotlib.pyplot as plt
from model_lstm_v1 import build_lstm_model
from model_gru import build_gru_model
from custom_callbacks import ThresholdCallback
from tensorflow import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from keras_tuner import RandomSearch
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def train_lstm(X_train, X_val, y_train, y_val):
    logging.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")

    # Build models
    epochs = 10
    batch_size = 64
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    threshold_callback = ThresholdCallback(threshold=70.0)  # threshold is set as 1.0

    # LSTM model
    lstm_model = build_lstm_model((X_train.shape[1], 1))
    lstm_history = lstm_model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=batch_size, 
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
    plot_filename = 'lstm_loss.png'
    plt.savefig(plot_filename)
    logging.info(f"Loss plot saved as {plot_filename}")
    plt.close()

    # Save model weights
    try:
        lstm_model.save_weights('lstm_01114_0900pm.weights.h5')
    except Exception as e:
        logging.error(f"Error saving model weights: {e}")

def lstm_predict(X_test):
    lstm_model = build_lstm_model((X_test.shape[1], 1))
    # Load trained weights (assuming you save them after training)
    lstm_model.load_weights('lstm_01114_0900pm.weights.h5')

    lstm_preds = lstm_model.predict(X_test)
    return lstm_preds

