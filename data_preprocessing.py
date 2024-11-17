# Description: This file contains functions to process CGM data, create sequences, and normalize data.
# Author: Balaji Anna
# Last Edited: 2024-Nov-17

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import configparser
import random

# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()

# Check if the config file exists
if not os.path.exists(config_file):
    raise FileNotFoundError(f"Config file '{config_file}' not found.")

# Read the config file
config.read(config_file)

# Get parameters from config file with error handling
try:
    LOG_PATH = config['LOG']['log_path']
except KeyError as e:
    print(f"Error: Missing key in config file: {e}")
    print("Available sections:", config.sections())
    print("Available keys in LOG section:", config['LOG'].keys() if 'LOG' in config.sections() else "No LOG section")
    raise

# Ensure the directory for the log file exists
log_dir = os.path.dirname(LOG_PATH)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Process subjects in the given directory, normalize data and return the processed data as X and y
def process_subjects(directory, num_subjects, samples_per_subject, time_window):
    logging.info(f"Processing data for {num_subjects} subjects and {samples_per_subject} samples per subjet in directory: {directory}")
    all_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if len(all_files) < num_subjects:
        logging.warning(f"Not enough files in directory. Using all {len(all_files)} files.")
        num_subjects = len(all_files)
    selected_files = random.sample(all_files, num_subjects)

    all_data = []    
    for filename in selected_files:
        file_path = os.path.join(directory, filename)
        logging.info(f"Processing file: {file_path}")
        try:
            cgm_data = process_cgm_data(file_path)
            if cgm_data is not None and len(cgm_data) >= samples_per_subject + time_window:
                start_index = random.randint(0, len(cgm_data) - samples_per_subject - time_window)
                selected_data = cgm_data['mg/dl'].values[start_index:start_index + samples_per_subject + time_window]
                all_data.append(selected_data)
            else:
                logging.warning(f"Insufficient data in {file_path}. Skipping.")
        except Exception as e:
            logging.warning(f"Error processing file {file_path}: {str(e)}")
            continue

    if not all_data:
        logging.error("No valid data found. Exiting.")
        return None, None, None

    combined_data = np.concatenate(all_data)
    logging.info(f"Combined data shape: {combined_data.shape}")

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data.reshape(-1, 1)).flatten()
    logging.info(f"Normalized data shape: {normalized_data.shape}")

    X, y = create_sequences(normalized_data, time_window)
    logging.info(f"X shape: {X.shape}")
    logging.info(f"y shape: {y.shape}")

    return X, y, scaler

# process CGM data from file for each subject and return the processed data as a DataFrame
def process_cgm_data(file_path):
    logging.info(f"Processing CGM data from file: {file_path}")
    try:
        df = pd.read_excel(file_path, sheet_name='CGM')
        df['date'] = pd.to_datetime(df['date']).dt.floor('min')
        if 'mg/dl' not in df.columns:
            df.columns = ['date', 'mg/dl']
        df = df.dropna()
        logging.info(f"Data processed for CGM data in {file_path}")
        return df
    except Exception as e:
        logging.warning(f"Error processing CGM data in {file_path}: {str(e)}")
        return None

# Create sequences from the given data with the specified time window
def create_sequences(data, time_window):
    logging.info(f"Creating sequences with time window: {time_window}")
    X, y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:(i + time_window)])
        y.append(data[i + time_window])
    logging.info(f"Number of sequences created: {len(X)}")
    return np.array(X), np.array(y)
