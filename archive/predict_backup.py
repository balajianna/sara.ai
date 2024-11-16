# Description: This script is used to make glucose predictions using a trained model
# Author: Balaji Anna
# Last Edited: 2024-Nov-16

import logging
import numpy as np
from tensorflow import keras
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from data_preprocessing import process_cgm_data
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

import configparser
# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Get parameters from config file
TIME_WINDOW = config.getint('DATA', 'time_window')
DATA_DIRECTORY = config['DATA']['directory']
MODEL_PATH = config['MODEL']['keras_model_path']
SCALER_PATH = config['MODEL']['scaler_path']

# Set up logging
logging.basicConfig(
    filename=f'glucose_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = keras.models.load_model(model_path, compile=False)  # Don't compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')  # Recompile with default optimizer
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        return None, None

def preprocess_input(input_data, scaler, time_window):
    try:
        input_data = np.array(input_data).reshape(-1, 1)
        scaled_data = scaler.transform(input_data)
        return scaled_data.reshape(1, time_window, 1)
    except Exception as e:
        logging.error(f"Error preprocessing input: {e}")
        return None

def predict_glucose(model, scaler, input_data):
    try:
        prediction = model.predict(input_data)
        original_prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        return original_prediction[0][0]
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None
    
def plot_results(actual, predicted, timestamps, title):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, actual, label='Actual', marker='o')
    plt.plot(timestamps, predicted, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.close()

def file_prediction(model, scaler, time_window, data_directory):
    try:
        print("\nEnter the Subject number (e.g.: 31):")
        filename = 'Subject'+input().strip()+'.xlsx'
        file_path = os.path.join(data_directory, filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        # Process CGM data
        cgm_data = process_cgm_data(file_path)
        if cgm_data is None:
            print("Error processing CGM data.")
            return
        
        glucose_values = cgm_data['mg/dl'].values
        timestamps = cgm_data['date'].values
        
        print(f"Total glucose values in the file: {len(glucose_values)}")
        print(f"Number of predictions to be made: {len(glucose_values) - time_window}")
        
        # Scale all glucose values at once
        scaled_glucose = scaler.transform(glucose_values.reshape(-1, 1)).flatten()
        
        # Prepare input data
        X = np.array([scaled_glucose[i:i+time_window] for i in range(len(scaled_glucose) - time_window)])
        X = X.reshape(X.shape[0], time_window, 1)
        
        # Make predictions in batches
        batch_size = 1024  # Adjust based on your system's memory
        predictions = []
        
        for i in tqdm(range(0, len(X), batch_size), desc="Processing"):
            batch = X[i:i+batch_size]
            batch_predictions = model.predict(batch, verbose=0)
            predictions.extend(batch_predictions)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        actual_values = glucose_values[time_window:]
        prediction_timestamps = timestamps[time_window:]
        
        # Plot results
        plot_results(actual_values, predictions, prediction_timestamps, "Glucose Prediction Results")
        print(f"Prediction results plotted and saved as glucose_prediction_results.png")
        
        # Save predictions to Excel
        results_df = pd.DataFrame({
            'Timestamp': prediction_timestamps,
            'Actual': actual_values,
            'Predicted': predictions
        })
        results_filename = f'prediction_results_{filename}'
        results_df.to_excel(results_filename, index=False)
        print(f"Prediction results saved to {results_filename}")
        
    except Exception as e:
        logging.error(f"Error in file prediction: {e}")
        print(f"An error occurred: {e}")

def main(model_path, scaler_path, time_window, data_directory):
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        print("Failed to load model or scaler. Exiting.")
        return
    
    file_prediction(model, scaler, time_window, data_directory)

if __name__ == "__main__":
    main(MODEL_PATH, SCALER_PATH, TIME_WINDOW, DATA_DIRECTORY)