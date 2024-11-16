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

def train_ensemble(X_train, X_val, y_train, y_val):
    logging.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")

    # Build models
    epochs = 4
    batch_size = 64
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    threshold_callback = ThresholdCallback(threshold=0.1)  # threshold is set as 0.1

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

    # GRU model
    gru_model = build_gru_model((X_train.shape[1], 1))
    gru_history = gru_model.fit(
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
    plt.plot(gru_history.history['loss'], label='GRU Training Loss')
    plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
    plt.title('Model Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Save plot
    plot_filename = f'ensemble_loss_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_filename)
    logging.info(f"Loss comparison plot saved as {plot_filename}")
    plt.close()

    # Save model weights
    lstm_model.save_weights(f'lstm_weights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
    gru_model.save_weights(f'gru_weights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')

def ensemble_predict(X_test):
    lstm_model = build_lstm_model((X_test.shape[1], 1))
    gru_model = build_gru_model((X_test.shape[1], 1))

    # Load trained weights (assuming you save them after training)
    lstm_model.load_weights('lstm_weights_latest.h5')
    gru_model.load_weights('gru_weights_latest.h5')

    lstm_preds = lstm_model.predict(X_test)
    gru_preds = gru_model.predict(X_test)

    # Average predictions from both models
    ensemble_preds = (lstm_preds + gru_preds) / 2

    return ensemble_preds
