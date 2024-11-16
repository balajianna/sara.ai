import logging
import numpy as np
from tensorflow import keras
import joblib
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'glucose_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        return None, None

def preprocess_input(input_data, scaler, time_window):
    try:
        # Ensure input_data is a 2D array
        input_data = np.array(input_data).reshape(-1, 1)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Reshape for LSTM input (assuming single feature)
        return scaled_data.reshape(1, time_window, 1)
    except Exception as e:
        logging.error(f"Error preprocessing input: {e}")
        return None

def predict_glucose(model, scaler, input_data):
    try:
        prediction = model.predict(input_data)
        # Inverse transform the prediction
        original_prediction = scaler.inverse_transform(prediction)
        return original_prediction[0][0]
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None

def main(model_path, scaler_path, time_window):
    model_path = model_path
    scaler_path = scaler_path
    time_window = time_window

    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if model is None or scaler is None:
        print("Failed to load model or scaler. Exiting.")
        return

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

if __name__ == "__main__":
    model_path = 'path_to_your_saved_model.h5'
    scaler_path = 'path_to_your_saved_scaler.joblib'
    time_window = 60  # Should match the TIME_WINDOW used in training
    main()