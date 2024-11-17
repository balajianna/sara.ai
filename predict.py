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
import configparser
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Get parameters from config file
TIME_WINDOW = config.getint('DATA', 'time_window')
DATA_DIRECTORY = config['DATA']['directory']
MODEL_PATH = config['MODEL']['keras_model_path']
SCALER_PATH = config['MODEL']['scaler_path']
SUBJECT_IDS = config['PREDICT']['subject_ids'].split(',')

OUTPUT_DIR = config['LOG']['output_dir']

# Set up logging
logging.basicConfig(
    filename=f'glucose_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')
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

    plot_result_filename = f'{title.replace(" ", "_").lower()}.png'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, plot_result_filename))
    plt.close()
    logging.info(f"Plat results  saved as {plot_result_filename}")

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return mse, rmse, mae, r2

def batch_prediction(model, scaler, time_window, data_directory, subject_ids):
    for subject_id in subject_ids:
        try:
            filename = f'Subject{subject_id.strip()}.xlsx'
            file_path = os.path.join(data_directory, filename)
            
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}")
                continue

            # Process CGM data
            cgm_data = process_cgm_data(file_path)
            if cgm_data is None:
                logging.error(f"Error processing CGM data for Subject {subject_id}")
                continue

            glucose_values = cgm_data['mg/dl'].values
            timestamps = cgm_data['date'].values

            logging.info(f"Subject {subject_id}: Total glucose values - {len(glucose_values)}")
            logging.info(f"Subject {subject_id}: Number of predictions - {len(glucose_values) - time_window}")

            # Scale all glucose values at once
            scaled_glucose = scaler.transform(glucose_values.reshape(-1, 1)).flatten()

            # Prepare input data
            X = np.array([scaled_glucose[i:i+time_window] for i in range(len(scaled_glucose) - time_window)])
            X = X.reshape(X.shape[0], time_window, 1)

            # Make predictions in batches
            batch_size = 1024
            predictions = []
            for i in tqdm(range(0, len(X), batch_size), desc=f"Processing Subject {subject_id}"):
                batch = X[i:i+batch_size]
                batch_predictions = model.predict(batch, verbose=0)
                predictions.extend(batch_predictions)

            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            actual_values = glucose_values[time_window:]
            prediction_timestamps = timestamps[time_window:]

            # Plot results
            plot_results(actual_values, predictions, prediction_timestamps, f"Glucose Prediction Results - Subject {subject_id}")
            logging.info(f"Subject {subject_id}: Prediction results plotted and saved")

            # Calculate metrics
            mse, rmse, mae, r2 = calculate_metrics(actual_values, predictions)
            logging.info(f"Subject {subject_id} Metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

            # Save predictions to Excel
            results_df = pd.DataFrame({
                'Timestamp': prediction_timestamps,
                'Actual': actual_values,
                'Predicted': predictions
            })

            filename = f'prediction_results_Subject{subject_id}.xlsx'
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            results_filename = os.path.join(OUTPUT_DIR, filename)
            results_df.to_excel(results_filename, index=False)
            logging.info(f"Subject {subject_id}: Prediction results saved to {results_filename}")

        except Exception as e:
            logging.error(f"Error in prediction for Subject {subject_id}: {e}")

def main(model_path, scaler_path, time_window, data_directory, subject_ids):
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        logging.error("Failed to load model or scaler. Exiting.")
        return

    batch_prediction(model, scaler, time_window, data_directory, subject_ids)

if __name__ == "__main__":
    main(MODEL_PATH, SCALER_PATH, TIME_WINDOW, DATA_DIRECTORY, SUBJECT_IDS)