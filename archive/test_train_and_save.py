import os
import unittest
import logging
import numpy as np
import pandas as pd
from train_and_save_backup import train_and_save
from data_preprocessing import process_subjects
from train_lstm_backup import build_lstm_model
from predict import predict_glucose, load_model_and_scaler

class TestGlucosePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.data_dir = 'data'
        cls.create_mock_data()

    @classmethod
    def create_mock_data(cls):
        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5T')
        glucose_values = np.random.randint(70, 180, size=1000)
        df = pd.DataFrame({'date': dates, 'mg/dl': glucose_values})
        df.to_excel(os.path.join(cls.data_dir, 'test_subject.xlsx'), sheet_name='CGM', index=False)

    def test_data_preprocessing(self):
        X, y, scaler = process_subjects(self.data_dir, 1, 60)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(scaler)

    def test_model_building(self):
        model = build_lstm_model((60, 1), [64, 32], 0.2)
        self.assertIsNotNone(model)

    def test_train_and_save(self):
        model_file, scaler_file = train_and_save()
        self.assertIsNotNone(model_file, "Model file was not returned")
        self.assertIsNotNone(scaler_file, "Scaler file was not returned")
        
        if model_file and scaler_file:
            self.assertTrue(os.path.exists(model_file), f"Model file {model_file} does not exist")
            self.assertTrue(os.path.exists(scaler_file), f"Scaler file {scaler_file} does not exist")
            self.assertGreater(os.path.getsize(model_file), 0, "Model file is empty")
            self.assertGreater(os.path.getsize(scaler_file), 0, "Scaler file is empty")
            
            # Test loading and prediction
            model, scaler = load_model_and_scaler(model_file, scaler_file)
            self.assertIsNotNone(model)
            self.assertIsNotNone(scaler)
            
            mock_input = np.random.randint(70, 180, size=(1, 60, 1))
            prediction = predict_glucose(model, scaler, mock_input)
            self.assertIsNotNone(prediction)
            self.assertTrue(70 <= prediction <= 180)

            # Clean up
            os.remove(model_file)
            os.remove(scaler_file)
        else:
            self.fail("train_and_save returned None values")

    def test_error_handling(self):
        X, y, scaler = process_subjects('non_existent_directory', 1, 60)
        self.assertIsNone(X)
        self.assertIsNone(y)
        self.assertIsNone(scaler)

if __name__ == '__main__':
    unittest.main(verbosity=2)