import os
import unittest
import matplotlib.pyplot as plt
import numpy as np
from train_lstm_v1 import train_lstm, lstm_predict
from data_preprocessing import process_subjects
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename=f'glucose_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories and parameters
directory = 'data'  # Folder containing subject data files
time_window = 60  # Reduced time window for sequences (60 minutes)

def main():
    """
    Main function to run the entire pipeline: data preprocessing, training, testing, and evaluation.
    """
    # Step 1: Data Preprocessing
    logging.info("Preprocessing data for 25 subjects...")
    X, y, scaler = process_subjects(directory, 25, time_window)

    if X is None or y is None:
        logging.error("Data processing failed. Exiting.")
        return

    # Step 2: Split data into training, validation, and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.32, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.375, random_state=42)

    logging.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Step 3: Train lstm model
    logging.info("Training lstm model...")
    train_lstm(X_train[:, :, np.newaxis], X_val[:, :, np.newaxis], y_train, y_val)

    # Step 4: Evaluate on test set
    logging.info("Evaluating lstm predictions on test set...")
    lstm_predicted = lstm_predict(X_test[:, :, np.newaxis])

    # Step 5: Plot actual vs predicted blood glucose levels
    plt.figure(figsize=(12, 6))
    y_pred_rescaled = scaler.inverse_transform(lstm_predicted.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.plot(y_test_rescaled.flatten(), label='Actual', alpha=0.7)
    plt.plot(y_pred_rescaled.flatten(), label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Blood Glucose Levels')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.xlabel('Time')
    plt.legend()
    
    # Step 6:Save plot
    plot_filename = 'actual_vs_predicted.png'
    plt.savefig(plot_filename)
    logging.info(f"Plot saved as {plot_filename}")
    plt.close()

if __name__ == "__main__":
    main()