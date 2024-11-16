import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from data_preprocessing import process_subjects
from train_lstm import train_lstm, lstm_predict, get_config, load_model_and_scaler
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(
    filename=f'glucose_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define directories and parameters
DIRECTORY = 'data'  # Folder containing subject data files
TIME_WINDOW = 60  # Time window for sequences (60 minutes)
NUM_SUBJECTS = 10  # Number of subjects to process


def main():
    """
    Main function to run the entire pipeline: data preprocessing, training, testing, and evaluation.
    """
    try:
        # Step 1: Data Preprocessing
        logging.info(f"Preprocessing data for {NUM_SUBJECTS} subjects...")
        X, y, scaler = process_subjects(DIRECTORY, NUM_SUBJECTS, TIME_WINDOW)
        if X is None or y is None:
            raise ValueError("Data processing failed.")
        
        # Step 2: Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.35, random_state=42)
        logging.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get configuration
        config = get_config()
        logging.info(f"Configuration: {config}")
        
        # Step 3: Train LSTM model
        logging.info("Training LSTM model...")
        lstm_model, model_filename, scaler_filename = train_lstm(
            X_train[:, :, np.newaxis],
            X_val[:, :, np.newaxis],
            y_train,
            y_val,
            config,
            scaler
        )
        
        # Step 4: Evaluate on test set
        logging.info("Evaluating LSTM predictions on test set...")
        lstm_predicted = lstm_predict(X_test[:, :, np.newaxis], lstm_model, scaler)
        
        # Step 5: Plot actual vs predicted blood glucose levels
        plot_results(y_test, lstm_predicted, scaler)
        
        # Example of loading the model and predicting
        try:
            loaded_model, loaded_scaler = load_model_and_scaler(model_filename, scaler_filename)
            
            # Make predictions using the loaded model
            logging.info("Evaluating LSTM predictions on test set using loaded model...")
            lstm_predicted_loaded = lstm_predict(X_test[:, :, np.newaxis], loaded_model, loaded_scaler)
            
            # Plot actual vs predicted blood glucose levels using loaded model
            plot_results(y_test, lstm_predicted_loaded, loaded_scaler, suffix="_loaded_model")
            
        except Exception as e:
            logging.error(f"An error occurred during prediction with loaded model: {e}")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def plot_results(y_test, y_pred, scaler):
    """
    Plot and save the actual vs predicted blood glucose levels.
    """
    plt.figure(figsize=(12, 6))
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    plt.plot(y_test_rescaled.flatten(), label='Actual', alpha=0.7)
    plt.plot(y_pred_rescaled.flatten(), label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Blood Glucose Levels')
    plt.ylabel('Blood Glucose (mg/dL)')
    plt.xlabel('Time')
    plt.legend()

    # Step 6: Save plot
    plot_filename = f'actual_vs_predicted_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(plot_filename)
    logging.info(f"Plot saved as {plot_filename}")
    plt.close()

if __name__ == "__main__":
    main()