from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import keras_tuner as kt
import os, sys, math, warnings, shutil
from data_processing_new import get_dates_five_years, get_trading_dates, get_dataset, inverse_transform_data, get_transformed_final_window


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
        lstm_hypermodel.add((keras.layers.LSTM(units=units, input_shape=(time_steps, features), return_sequences=True, recurrent_dropout=dropout)))
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
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')

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
        dropout = 0.6
    else:
        layers = hps['layers']
        units = hps['units']
        dropout = hps['dropout']

    lstm_model = keras.models.Sequential()

    for _ in range(layers):
        lstm_model.add((keras.layers.LSTM(units=units, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=dropout)))
    lstm_model.add(keras.layers.Dense(units=1, activation='linear'))

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    print_train_progress_callback = CustomCallback(epochs, window)

    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_x, train_y, epochs=epochs, validation_split=0.25, verbose=0, callbacks=[early_stopping_callback, print_train_progress_callback])

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


def experiment(stock_ticker, time_steps, date_range=None, drop_col=None, test_on_val=False, hps=None, window=None):
    """Function that creates and evaluates a single model.
    Returns the performance metrics of the created model.

    Args:
        stock_ticker (string): The target stock to be predicted.
        time_steps (int): The number of timesteps in the data window inputs.
        epochs (int): The maximum number of training epochs.

    Returns:
        dict: A dictionary of the performance metrics of the created model.
    """

    linear_scaler, scaler, col_names, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=date_range, time_steps=time_steps, drop_col=drop_col)

    if test_on_val:
        test_len = train_x.shape[0] * 25 // 100
        test_x = train_x[-test_len:]
        test_y = train_y[-test_len:]

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x, train_y, epochs=100, hps=hps, window=window)

    if window is not None:
        window.ui.train_progress_label.setText('100.0 %')

    # get the model predictions
    predictions = forecast_lstm_model(lstm_model, test_x, window=window)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y = np.array([test_y[i, -1] for i in range(len(test_y))])

    # revert the normalization scalings done
    test_y = inverse_transform_data(test_y, scaler, col_names, feature="log_return")
    predictions = inverse_transform_data(predictions, scaler, col_names, feature="log_return")

    # get model performance statistics
    perf = get_lstm_model_perf(predictions, test_y)

    return perf, lstm_model, linear_scaler, scaler, col_names





def make_model_forecast(model_dict, final_window):

    model = model_dict['model']
    scaler = model_dict['scaler']
    col_names = model_dict['col_names']

    final_prediction = forecast_lstm_model(model, final_window)
    final_prediction = inverse_transform_data(final_prediction, scaler, col_names, feature="log_return")

    final_prediction = [i.tolist() for i in final_prediction][0]
    final_prediction = 1 if final_prediction >= 0 else 0

    return final_prediction





def feature_selection(stock_ticker, timesteps, date_range=None, repeats=20, hps=None):
    
    features = ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    num_features = len(features)
    dropped_features = features.copy()

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current: None)")
    

    model_perfs = []
    for _ in range(repeats):
        curr_model_perf, _, _, _, _ = experiment(stock_ticker, timesteps, date_range=date_range, drop_col=dropped_features, test_on_val=True, hps=hps)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = mean(model_perfs)
    
    print(f"Current Mean Directional Accuracy: {round(curr_best_da, 6)}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")


    for index, feature in enumerate(features):

        model_perfs = []
        dropped_features.remove(feature)

        print(f"Features Tested: {index + 1}/{num_features} (current features: {[feature for feature in features if feature not in dropped_features]})")

        for _ in range(repeats):
            curr_model_perf, _, _, _, _ = experiment(stock_ticker, timesteps, date_range=date_range, drop_col=dropped_features, test_on_val=True, hps=hps)
            model_perfs.append(curr_model_perf['da'])

        curr_da = mean(model_perfs)

        if curr_da > curr_best_da:
            curr_best_da = curr_da
        else:
            dropped_features.append(feature)

        print(f"Best Mean Directional Accuracy: {round(curr_best_da, 6)}")
        print(f"Current Mean Directional Accuracy: {round(curr_da, 6)}")
        
        print(f"Dropped Features: {dropped_features}")
        print("===================================================")

    return dropped_features


def get_params(stock_ticker, date_range=None):

    time_steps_list = [10, 15]
    dropped_features = []

    for step in time_steps_list:
        curr_dropped_features = feature_selection(stock_ticker, step, date_range=date_range, repeats=15, hps=None)
        dropped_features.append(curr_dropped_features)

    hps_list = []

    for index, time_steps in enumerate(time_steps_list):
        _, _, _, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=date_range, time_steps=time_steps, drop_col=dropped_features[index])
        hps = get_optimal_hps(train_x, train_y)
        hps_list.append(hps)


    print(dropped_features)
    print(hps_list)


    repeats = 10
    average_performances = [0] * len(time_steps_list)

    for index in range(len(time_steps_list)):
        print("===================================================")
        performances = []

        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")
            perf, _, _, _, _ = experiment(stock_ticker, time_steps_list[index], date_range=date_range, drop_col=dropped_features[index], hps=hps_list[index])
            performances.append(perf)
            print("===================================================")

        mean_da = mean([perf['da'] for perf in performances])
        mean_uda = mean([perf['uda'] for perf in performances])
        mean_dda = mean([perf['dda'] for perf in performances])

        std_da = stdev([perf['da'] for perf in performances])
        std_uda = stdev([perf['uda'] for perf in performances])
        std_dda = stdev([perf['dda'] for perf in performances])
        
        optimistic_baseline = performances[0]['total_ups'] / (performances[0]['total_ups'] + performances[0]['total_downs'])
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

        average_performances[index] = mean_da


    print(dropped_features)
    print(hps_list)


    best_hps_index = average_performances.index(max(average_performances))

    best_dropped_features = dropped_features[best_hps_index]
    best_hps = hps_list[best_hps_index]
    best_time_steps = time_steps_list[best_hps_index]

    print('===========================')

    for i in dropped_features:
        print(i)

    for i in hps_list:
        print(i)

    print('===========================')

    return {
        'dropped_features': best_dropped_features,
        'hps': best_hps,
        'time_steps': best_time_steps
    }



def main():
    # stock to be predicted
    stock_ticker = 'PGOLD'

    # parameters of each model
    time_steps = 1

    hps = {'units': 32, 'layers': 1, 'dropout': 0.30000000000000004, 'tuner/epochs': 4, 'tuner/initial_epoch': 0, 'tuner/bracket': 3, 'tuner/round': 0}

    # how many models built (min = 2)
    repeats = 10

    # dropped features
    dropped_features = ['wr', 'cmf', 'rsi', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'inflation', 'real_interest_rate', 'roe', 'psei_returns', 'sentiment']

    
    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _, _, _ = experiment(stock_ticker, time_steps, date_range=None, drop_col=dropped_features, hps=hps)
        performances.append(perf)
        print("===================================================")

    mean_da = mean([perf['da'] for perf in performances])
    mean_uda = mean([perf['uda'] for perf in performances])
    mean_dda = mean([perf['dda'] for perf in performances])

    std_da = stdev([perf['da'] for perf in performances])
    std_uda = stdev([perf['uda'] for perf in performances])
    std_dda = stdev([perf['dda'] for perf in performances])
    
    optimistic_baseline = performances[0]['total_ups'] / (performances[0]['total_ups'] + performances[0]['total_downs'])
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



def test_forecast():
    stock_ticker = 'JFC'

    time_steps = 1
    hps = None
    dropped_features = None

    perf, model, linear_scaler, scaler, col_names = experiment(stock_ticker, time_steps, date_range=None, drop_col=dropped_features, hps=hps)

    print(perf)

    params = {
        'time_steps': time_steps,
        'dropped_features': dropped_features
    }

    model_dict = {
        'model': model,
        'linear_scaler': linear_scaler,
        'scaler': scaler,
        'col_names': col_names,
        'params': params

    }

    final_window, last_observed_trading_day = get_transformed_final_window(stock_ticker, model_dict)

    for i in range(2):
        print(make_model_forecast(model_dict, final_window))


if __name__ == '__main__':

    warnings.simplefilter('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    main()

    # hps = {'units': 128, 'layers': 1, 'dropout': 0.0, 'tuner/epochs': 34, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}

    # stock_ticker = 'PGOLD'
    # params = get_params(stock_ticker)
    # print(params)

    # test_forecast()
