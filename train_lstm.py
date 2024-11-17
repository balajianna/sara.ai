# Description: This script is used to train the LSTM model for the final project submission
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
import configparser
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from model_lstm import build_lstm_model
import os
 
# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# For dynamic parameter seletion, load values for hyperparameter tuning from config.ini
USE_HYPERPARAMETER_TUNING = config.getboolean('HYPER', 'use_hyperparameter_tuning')
NUMBER_OF_SUBJECTS_FOR_TUNING = config.getint('HYPER', 'number_of_subjects_for_parameter_tuning')
SAMPLES_PER_SUBJECT_FOR_TUNING = config.getint('HYPER', 'samples_per_subject_for_parameter_tuning')
EPOCHS_FOR_TUNING = config.getint('HYPER', 'epochs_for_parameter_tuning')
BATCH_SIZE_FOR_TUNING = config.getint('HYPER', 'batch_size_for_parameter_tuning') 
NUMBER_OF_TRIALS_FOR_TUNING = config.getint('HYPER', 'number_of_trials_for_parameter_tuning')

# To select predefined model architecture and prameters, Load  config.ini
EPOCHS = config.getint('TRAINING', 'epochs')
BATCH_SIZE = config.getint('TRAINING', 'batch_size')
THRESHOLD = config.getfloat('TRAINING', 'threshold')
DROPOUT_RATE = config.getfloat('TRAINING', 'dropout_rate')
LSTM_LAYERS = eval(config.get('TRAINING', 'lstm_layers'))
OPTIMIZER = config.get('TRAINING', 'optimizer')
LEARNING_RATE = config.getfloat('TRAINING', 'learning_rate')
EARLY_STOPPING_PATIENCE = config.getint('TRAINING', 'early_stopping_patience')
REDUCE_LR_PATIENCE = config.getint('TRAINING', 'reduce_lr_patience')
LOSS = config.get('TRAINING', 'loss')
METRICS = config.get('TRAINING', 'metrics')
DENSE_UNIT = config.getint('TRAINING', 'dense_unit')
ACTIVATION = config.get('TRAINING', 'activation')

OUTPUT_DIR = config.get('LOG', 'output_dir')

# Objective function for hyperparameter tuning
def objective(trial, X_train, X_val, y_train, y_val):
    config = {
        'lstm_layers': [
            trial.suggest_int(f'lstm_units_{i}', 32, 256)
            for i in range(trial.suggest_int('n_lstm_layers', 1, 3))
        ],
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }

    model = build_lstm_model(
        input_shape=(X_train.shape[1], 1),
        lstm_layers=config['lstm_layers'],
        dropout_rate=config['dropout_rate']
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss=LOSS,
        metrics=[METRICS]
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS_FOR_TUNING,
        batch_size=config['batch_size'],
        validation_data=(X_val, y_val),
        verbose=0
    )

    return history.history['val_loss'][-1]

# Find the best hyperparameters using Optuna
def find_best_params(X_train, X_val, y_train, y_val, n_trials=NUMBER_OF_TRIALS_FOR_TUNING):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=n_trials)
    plot_hyperparameter_importance(study)
    plot_parameter_relationships(study)
    return study.best_params, study.best_value

# Train the LSTM model
def train_lstm(X_train, X_val, y_train, y_val, use_hyperparameter_tuning=USE_HYPERPARAMETER_TUNING):
    logging.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
    logging.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")

    try:
        if use_hyperparameter_tuning:
            try:
                logging.info("Tuning hyperparameters with a subset of data...")
                X_tune = X_train[:NUMBER_OF_SUBJECTS_FOR_TUNING * SAMPLES_PER_SUBJECT_FOR_TUNING]
                y_tune = y_train[:NUMBER_OF_SUBJECTS_FOR_TUNING * SAMPLES_PER_SUBJECT_FOR_TUNING]
                X_tune_val = X_val[:NUMBER_OF_SUBJECTS_FOR_TUNING * SAMPLES_PER_SUBJECT_FOR_TUNING]
                y_tune_val = y_val[:NUMBER_OF_SUBJECTS_FOR_TUNING * SAMPLES_PER_SUBJECT_FOR_TUNING]

                best_params, best_value = find_best_params(X_tune, X_tune_val, y_tune, y_tune_val)
                logging.info(f"Best hyperparameters: {best_params}")
                logging.info(f"Best validation loss: {best_value}")

                lstm_layers = [best_params[f'lstm_units_{i}'] for i in range(best_params['n_lstm_layers'])]
                dropout_rate = best_params['dropout_rate']
                learning_rate = best_params['learning_rate']
                batch_size = best_params['batch_size']
            except Exception as e:
                logging.error(f"Error in hyperparameter tuning: {str(e)}")
                logging.info("Using predefined model architecture from config...")
                lstm_layers = LSTM_LAYERS
                dropout_rate = DROPOUT_RATE
                learning_rate = LEARNING_RATE
                batch_size = BATCH_SIZE
        else:
            logging.info("Using predefined model architecture from config...")
            lstm_layers = LSTM_LAYERS
            dropout_rate = DROPOUT_RATE
            learning_rate = LEARNING_RATE
            batch_size = BATCH_SIZE

        # Build model using the best or predefined hyperparameters
        lstm_model = build_lstm_model(
            input_shape=(X_train.shape[1], 1),
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        lstm_model.compile(optimizer=optimizer, loss=LOSS, metrics=[METRICS])

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6
        )

        threshold_callback = ThresholdCallback(threshold=THRESHOLD)

        history = lstm_model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, threshold_callback]
        )

        try:
            plot_training_history(history)
        except Exception as e:
            logging.error(f"Error in plotting training history: {str(e)}")
            return lstm_model, history

        try:
            y_pred = lstm_model.predict(X_val)
            mse, mae, rmse = calculate_metrics(y_val, y_pred)
            logging.info(f"Validation Metrics - MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        except Exception as e:
            logging.error(f"Error in calculating metrics: {str(e)}")
            return lstm_model, history

        return lstm_model, history

    except Exception as e:
        logging.error(f"Error in training LSTM model: {str(e)}")
        raise

# Plot the hyperparameter importance
def plot_hyperparameter_importance(study):
    importance = optuna.importance.get_param_importances(study)
    plt.figure(figsize=(10, 6))
    plt.bar(importance.keys(), importance.values())
    plt.title('Hyperparameter Importance')
    plt.xlabel('Hyperparameter')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()

    hyperparameter_importance_filename = f'hyperparameter_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, hyperparameter_importance_filename))
    logging.info(f"Hyperparameter importance plot saved as {hyperparameter_importance_filename}")   
    plt.close()

# Plot the relationship between hyperparameters
def plot_parameter_relationships(study):
    fig = optuna.visualization.plot_param_importances(study)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.write_image(os.path.join(OUTPUT_DIR, f'parameter_relationships_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))

# Plot the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()

    training_history_filename = f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, training_history_filename))
    logging.info(f"Training history plot saved as {training_history_filename}")
    plt.close()

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse
