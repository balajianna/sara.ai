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

def user_input_prediction(model, scaler, time_window):
    while True:
        try:
            print(f"\nEnter {time_window} glucose readings (one per line), or 'q' to quit:")
            input_data = []
            for i in range(time_window):
                value = input().strip()
                if value.lower() == 'q':
                    return
                input_data.append(float(value))
            
            processed_input = preprocess_input(input_data, scaler, time_window)
            if processed_input is None:
                print("Error processing input. Please try again.")
                continue
            
            prediction = predict_glucose(model, scaler, processed_input)
            if prediction is not None:
                print(f"Predicted glucose level: {prediction:.2f} mg/dL")
            else:
                print("Error making prediction. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print("An unexpected error occurred. Please try again.")



def file_prediction(model, scaler, time_window, data_directory):
    try:
        print("\nEnter the Susbject number (e.g.: 31):")
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
        
        actual_values = []
        predicted_values = []
        prediction_timestamps = []
        
        # Calculate the number of predictions to be made
        num_predictions = len(glucose_values) - time_window
        
        # Create batches for prediction
        batch_size = 100  # Adjust this value based on your system's capabilities
        
        with tqdm(total=num_predictions, desc="Processing") as pbar:
            for i in range(0, num_predictions, batch_size):
                batch_end = min(i + batch_size, num_predictions)
                batch_inputs = []
                
                for j in range(i, batch_end):
                    input_data = glucose_values[j:j+time_window]
                    processed_input = preprocess_input(input_data, scaler, time_window)
                    if processed_input is not None:
                        batch_inputs.append(processed_input)
                
                if batch_inputs:
                    batch_inputs = np.vstack(batch_inputs)
                    batch_predictions = model.predict(batch_inputs)
                    
                    for k, prediction in enumerate(batch_predictions):
                        actual_values.append(glucose_values[i+k+time_window])
                        predicted_values.append(prediction[0])
                        prediction_timestamps.append(timestamps[i+k+time_window])
                
                pbar.update(batch_end - i)
        
        # Plot results
        plot_results(actual_values, predicted_values, prediction_timestamps, "Glucose Prediction Results")
        print(f"Prediction results plotted and saved as glucose_prediction_results.png")
        
        # Save predictions to Excel
        results_df = pd.DataFrame({
            'Timestamp': prediction_timestamps,
            'Actual': actual_values,
            'Predicted': predicted_values
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

    """
    while True:
        print("\nChoose an option:")
        print("1. User input prediction")
        print("2. File-based prediction")
        print("3. Quit")
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            user_input_prediction(model, scaler, time_window)
        elif choice == '2':
            file_prediction(model, scaler, time_window, data_directory)
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")
    """
    

if __name__ == "__main__":
    model_path = 'lstm_model_20241115_153307.keras'
    scaler_path = 'scaler_20241115_153307.joblib'
    time_window = 60  # Should match the TIME_WINDOW used in training
    data_directory = 'data'  # Directory containing subject files
    main(model_path, scaler_path, time_window, data_directory)