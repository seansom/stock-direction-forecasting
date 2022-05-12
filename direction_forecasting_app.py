# random split with shuffling
from tensorflow import keras, compat
from statistics import mean, stdev
import keras_tuner as kt
import numpy as np
import pandas as pd
import os, sys, math, copy, shutil
from sklearn.preprocessing import PowerTransformer
from data_processing_app import get_dataset, inverse_transform_data, get_dates_five_years


class CustomCallback(keras.callbacks.Callback):
    """A callback class used to print the progress of model fitting
    after each epoch.
    """	
    def __init__(self, epochs, window=None):
        self.epochs = epochs
        self.window = window

    def on_epoch_end(self, epoch, logs=None):
        curr_progress = round(((epoch + 1) / self.epochs) * 100, 2)
        print(f'Training Progress: {curr_progress} %', end='\r')

        if self.window is not None:
            self.window.ui.train_progress_label.setText(f'{curr_progress} %')

    def on_train_end(self, logs=None):
        print()


def make_lstm_hypermodel(hp, time_steps, features):
    # a hypermodel has keras tuner hyperparameters (hp) that are variable
    lstm_hypermodel = keras.models.Sequential()

    # set hyperparameters to be searched in Hyperband tuning
    units = hp.Choice('units', values=[32, 64, 128, 256])
    layers = hp.Int('layers', min_value=1, max_value=5, step=1)
    dropout = hp.Float('dropout', min_value=0.0, max_value=0.9, step=0.1)

    # create lstm hypermodel
    for _ in range(layers):
        lstm_hypermodel.add(keras.layers.LSTM(units=units, input_shape=(time_steps, features), return_sequences=True,
                                              recurrent_dropout=dropout))
    lstm_hypermodel.add(keras.layers.Dense(units=1, activation="linear"))

    lstm_hypermodel.compile(loss='mean_squared_error', optimizer='adam')

    return lstm_hypermodel


def get_optimal_hps(train_x, train_y):
    # the tuner saves files to the current working directory, delete old files if any
    if os.path.exists('untitled_project'):
        shutil.rmtree('untitled_project')

    time_steps = train_x.shape[1]
    features = train_x.shape[2]

    # create a wrapper for the hypermodel builder to account for different input shapes
    hypermodel_builder = lambda hp: make_lstm_hypermodel(hp, time_steps, features)

    # if overwrite is false, previously-saved computed hps will be used
    tuner = kt.Hyperband(hypermodel_builder, objective='val_loss', max_epochs=100, factor=3, overwrite=True)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

    # execute Hyperband search of optimal hyperparameters
    tuner.search(train_x, train_y, validation_split=0.25, callbacks=[early_stopping_callback])

    # hps is a dictionary of optimal hyperparameter levels
    hps = (tuner.get_best_hyperparameters(num_trials=1)[0]).values.copy()

    # delete files saved by tuner in current working directory
    shutil.rmtree('untitled_project')

    return hps


def make_lstm_model(train_x, train_y, epochs=100, hps=None, window=None):

    if hps is None:
        layers = 3
        units = 64
        dropout = 0.2
    else:
        layers = hps['layers']
        units = hps['units']
        dropout = hps['dropout']

    lstm_model = keras.models.Sequential()

    for _ in range(layers):
        lstm_model.add(keras.layers.LSTM(units=units, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=dropout))
    lstm_model.add(keras.layers.Dense(units=1, activation='linear'))

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    print_train_progress_callback = CustomCallback(epochs, window)

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_x, train_y, epochs=epochs, validation_split=0.25,  verbose=0, callbacks=[early_stopping_callback, print_train_progress_callback])

    return lstm_model


def forecast_lstm_model(model, test_x, window=None):
    """Forecasts future values using a model and test input dataset.

    Args:
        model (Model): The built Keras model used for forecasting.
        test_x (np.array): The model inputs for testing and forecasting.

    Returns:
        np.array: A numpy array of the forecasted future values.
    """	
    predictions = []

    test_len = test_x.shape[0]
    test_timesteps = test_x.shape[1]
    test_features = test_x.shape[2]

    # Make a separate prediction for each test data window in the test dataset
    for i in range(test_len):
        curr_progress = round(((i + 1) / test_len) * 100, 2)
        print(f'Prediction Progress: {curr_progress} %', end='\r')

        if window is not None:
            window.ui.prediction_progress_label.setText(f'{curr_progress} %')

        model_input = (test_x[i, :, :]).reshape(1, test_timesteps, test_features)
        prediction = model.predict(model_input)
        # the prediction has a shape of (1, timesteps, 1)
        # only need to get the last prediction value
        predictions.append(prediction[0, -1, 0])
    print()

    predictions = (np.array(predictions)).flatten()
    return predictions


def get_lstm_model_perf(predictions, actuals):
    """Calculates performance metrics of a model given its predictions
    and actual future values.

    Args:
        predictions (np.array): A numpy array of forecasted future values.
        actuals (np.array): A numpy array of actual future values.

    Returns:
        dict: A dictionary containing difference performance metrics of a model.
    """	

    predictions_len = len(predictions)

    # calculate number of total actual upward and downward directions
    total_ups = sum([1 if actuals[i] >= 0 else 0 for i in range(len(actuals))])
    total_downs = len(actuals) - total_ups

    # calculate true positives, true negatives, false positives, and false negatives
    tp = sum([1 if (predictions[i] >= 0 and actuals[i] >= 0) else 0 for i in range(predictions_len)])
    tn = sum([1 if (predictions[i] < 0 and actuals[i] < 0) else 0 for i in range(predictions_len)])
    fp = sum([1 if (predictions[i] >= 0 and actuals[i] < 0) else 0 for i in range(predictions_len)])
    fn = sum([1 if (predictions[i] < 0 and actuals[i] >= 0) else 0 for i in range(predictions_len)])

    # calculate directional accuracy, upward directional accuracy, and downward directional accuracy
    da = (tp + tn) / (tp + tn + fp + fn)
    uda = (tp / (tp + fp)) if (tp + fp) else 1
    dda = (tn / (tn + fn)) if (tn + fn) else 1

    # store performance metrics in a dictionary
    return {"total_ups":total_ups, "total_downs":total_downs, "tp":tp, "tn":tn, "fp":fp, "fn":fn, "da":da, "uda":uda, "dda":dda}


def print_model_performance(perf):
    """Prints out the performance metrics of a model to the terminal.

    Args:
        perf (dict): A dictionary containing difference performance metrics of a model.
    """	

    print("===================================================")
    print(f"Total Ups: {perf['total_ups']}")
    print(f"Total Downs: {perf['total_downs']}")
    print("===================================================")
    print(f"TP: {perf['tp']}")
    print(f"TN: {perf['tn']}")
    print(f"FP: {perf['fp']}")
    print(f"FN: {perf['fn']}")
    print("===================================================")
    print(f"DA: {round(perf['da'], 6)}")
    print(f"UDA: {round(perf['uda'], 6)}")
    print(f"DDA: {round(perf['dda'], 6)}")
    print("===================================================")


def experiment(scaler, col_names, train_x, train_y, test_x, test_y, final_window, hps=None, window=None):

    train_x_copy = copy.deepcopy(train_x)
    train_y_copy = copy.deepcopy(train_y)
    test_x_copy = copy.deepcopy(test_x)
    test_y_copy = copy.deepcopy(test_y)

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x_copy, train_y_copy, epochs=100, hps=hps, window=window)

    # get the model predictions
    predictions = forecast_lstm_model(lstm_model, test_x_copy, window=window)

    final_prediction = forecast_lstm_model(lstm_model, final_window)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y_copy = np.array([test_y_copy[i, -1] for i in range(len(test_y_copy))])

    # revert the normalization scalings done
    test_y_copy = inverse_transform_data(test_y_copy, scaler, col_names, feature="log_return")
    predictions = inverse_transform_data(predictions, scaler, col_names, feature="log_return")
    final_prediction = inverse_transform_data(final_prediction, scaler, col_names, feature="log_return")
    final_prediction = [i.tolist() for i in final_prediction]

    # get model performance statistics
    perf = get_lstm_model_perf(predictions, test_y_copy)

    return lstm_model, perf, test_y_copy, predictions, final_prediction


def main():
    # stock to be predicted
    stock_ticker = 'AP'

    # parameters of each model
    time_steps = 20
    hps = None

    # how many models built (min = 2)
    repeats = 5

    
    print("===================================================")
    performances = []
    final_predictions = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        scaler, col_names, train_x, train_y, test_x, test_y, final_window = get_dataset(stock_ticker, date_range=get_dates_five_years(), time_steps=time_steps, drop_col=None)
        _, perf, _, _, final_prediction = experiment(scaler, col_names, train_x, train_y, test_x, test_y, final_window, hps)
        performances.append(perf)

        print(final_prediction)
        print(type(final_prediction))
        print('=======================')

        sys.exit()


        final_predictions.append(final_prediction)
        print("===================================================")

    mean_da = mean([perf['da'] for perf in performances])
    mean_uda = mean([perf['uda'] for perf in performances])
    mean_dda = mean([perf['dda'] for perf in performances])

    std_da = stdev([perf['da'] for perf in performances])
    std_uda = stdev([perf['uda'] for perf in performances])
    std_dda = stdev([perf['dda'] for perf in performances])

    mean_total_ups = mean([perf['total_ups'] for perf in performances])
    mean_total_downs = mean([perf['total_downs'] for perf in performances])
    
    optimistic_baseline = mean_total_ups / (mean_total_ups + mean_total_downs)
    pessimistic_baseline = 1 - optimistic_baseline

    print(f'Stock: {stock_ticker}')

    print()
    
    print(f'Total Ups: {performances[0]["total_ups"]}')
    print(f'Total Downs: {performances[0]["total_downs"]}')

    print()

    # Print average accuracies of the built models
    print(f"Mean DA: {round(mean_da, 6)}")
    print(f"Mean UDA: {round(mean_uda, 6)}")
    print(f"Mean DDA: {round(mean_dda, 6)}")

    print()

    print(f"Standard Dev. DA: {round(std_da, 6)}")
    print(f"Standard Dev. UDA: {round(std_uda, 6)}")
    print(f"Standard Dev. DDA: {round(std_dda, 6)}")

    print()

    print(f"Optimistic Baseline DA: {round(optimistic_baseline, 6)}")
    print(f"Pessimistic Baseline DA: {round(pessimistic_baseline, 6)}")

    print(f'Final Predictions: {final_predictions}')

main()