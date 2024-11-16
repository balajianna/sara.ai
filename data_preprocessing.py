# Description: This file contains functions to process CGM data, create sequences, and normalize data.
# Author: Balaji Anna
# Last Edited: 2024-Nov-16
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os
import logging
import configparser

# Read configuration file
config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(config_file)

# Get parameters from config file
DEBUG = config.getboolean('LOG', 'debug')

def process_cgm_data(file_path):
    if DEBUG:
        print("Processing CGM data from file:", file_path)
    
    try:
        df = pd.read_excel(file_path, sheet_name='CGM')
        df['date'] = pd.to_datetime(df['date']).dt.floor('min')
        if 'mg/dl' not in df.columns:
            df.columns = ['date', 'mg/dl']
        
        if DEBUG:
            print("Data processed for CGM data in", file_path)
            print(df.head())
        
        return df
    except Exception as e:
        logging.warning(f"Error processing CGM data in {file_path}: {str(e)}")
        return None

# Focus for Project Milestone
def process_subjects(directory, num_subjects, time_window):
    if DEBUG:
        print("Processing data for", num_subjects, "subjects in directory:", directory)

    all_data = []
    processed_count = 0

    for filename in os.listdir(directory):
        if processed_count >= num_subjects:
            break
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            if DEBUG:
                print("Processing file:", file_path)
            try:
                cgm_data = process_cgm_data(file_path)
                if cgm_data is not None:
                    if DEBUG:
                        print("cgm data is: ", cgm_data.head())
                    all_data.append(cgm_data['mg/dl'].values)
                    processed_count += 1
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {str(e)}")
                continue

    if not all_data:
        logging.error("No valid data found. Exiting.")
        return None

    combined_data = np.concatenate(all_data)
    if DEBUG:
        print("Combined data shape:", combined_data.shape)

    normalized_data = combined_data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data.reshape(-1, 1)).flatten()
    if DEBUG:
        print("Normalized data shape:", normalized_data.shape)

    X, y = create_sequences(normalized_data, time_window)
    if DEBUG:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
    
    return X, y, scaler

def create_sequences(data, time_window):
    if DEBUG:
        print("Creating sequences with time window:", time_window)

    X, y = [], []
    for i in range(len(data) - time_window):
        X.append(data[i:(i + time_window)])
        y.append(data[i + time_window])
    
    if DEBUG:
        print("Number of sequences created:", len(X))
        print("Sample sequences (X):", X[:5])
    return np.array(X), np.array(y)

# Added for expansion after Project milestone
def process_bolus_data(file_path):
    if DEBUG:
        print("Processing Bolus data from file:", file_path)

    try:
        df = pd.read_excel(file_path, sheet_name='Bolus')
        df['date'] = pd.to_datetime(df['date']).dt.floor('min')

        if DEBUG:
            print("Data processed for Bolus data in", file_path)
            print(df.head())
            
        return df
    except Exception as e:
        logging.warning(f"Error processing Bolus data in {file_path}: {str(e)}")
        return None

# Added for expansion after Project milestone
def process_basal_data(file_path):
    if DEBUG:
        print("Processing Basal data from file:", file_path)

    try:
        df = pd.read_excel(file_path, sheet_name='Basal')
        df['date'] = pd.to_datetime(df['date']).dt.floor('min')

        if DEBUG:
            print("Data processed for Basal data in", file_path)
            print(df.head())
        return df
    except Exception as e:
        logging.warning(f"Error processing Basal data in {file_path}: {str(e)}")
        return None
    
# Added for expansion after Project milestone
def process_subjects_CGM_Bolus_Basal(directory, num_subjects, time_window):
    if DEBUG:
        print("Processing data for", num_subjects, "subjects in directory:", directory)

    all_data = []
    processed_count = 0

    for filename in os.listdir(directory):
        if processed_count >= num_subjects:
            break
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            if DEBUG:
                print("Processing file:", file_path)
            try:
                cgm_data = process_cgm_data(file_path)
                if cgm_data is not None:
                    if DEBUG:
                        print("cgm data is: ", cgm_data.head())
                    all_data.append(cgm_data['mg/dl'].values)
                    processed_count += 1
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {str(e)}")
                continue
            
            # Added for expansion after Project milestone
            try:
                bolus_data = process_bolus_data(file_path)
                if bolus_data is not None:
                    if DEBUG:
                        print("bolus data is: ", bolus_data.head())
                    all_data.append(bolus_data['units'].values)
                    processed_count += 1    
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {str(e)}")
                continue
            
            try:                
                basal_data = process_basal_data(file_path)
                if basal_data is not None:
                    if DEBUG:
                        print("basal data is: ", basal_data.head())
                    all_data.append(basal_data['units'].values)
                    processed_count += 1
            except Exception as e:
                logging.warning(f"Error processing file {file_path}: {str(e)}")
                continue

    if not all_data:
        logging.error("No valid data found. Exiting.")
        return None

    combined_data = np.concatenate(all_data)
    if DEBUG:
        print("Combined data shape:", combined_data.shape)

    normalized_data = combined_data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(combined_data.reshape(-1, 1)).flatten()
    if DEBUG:
        print("Normalized data shape:", normalized_data.shape)

    X, y = create_sequences(normalized_data, time_window)
    if DEBUG:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
    
    return X, y, scaler
