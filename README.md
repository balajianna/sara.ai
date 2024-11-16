# sara
Stanford CS230 project repository

(1) config.ini contains all the parameteres needed

(2) custom_callbacks.py contains code to stop training if threshold is met

(3) data_preprocessing.py contains code for reading the subjects' file for training and validation, normalizing the glucose readings

(4) model_lstm.py contains model configuration. Parameters for model are set dynamically in another script

(5) train_lstm.py contains code for trains the model with selected subjects

(6) train_and_save.py contains code for (1) selecting the best parameters for hyper parameter if use_hyperparameter_tuning is enabled (2) if not, the training module reads default configuration of hyperparameters from config.ini (3) invokes train_lstm training method (4) The trained model and scalar is saved in order to allow for testing and prediction to be delinked and exeuted as many times as needed

(7) predict.py loads the saved model and evaluates the model by predicting user input subject files

