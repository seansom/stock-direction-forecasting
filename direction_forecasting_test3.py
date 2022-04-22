from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import keras_tuner as kt
import os, sys, math, warnings, shutil
from data_processing_old import get_dataset, inverse_transform_data


class CustomCallback(keras.callbacks.Callback):
    """A callback class used to print the progress of model fitting
    after each epoch.
    """	
    def __init__(self, epochs):
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        curr_progress = round(((epoch + 1) / self.epochs) * 100, 2)
        print(f'Training Progress: {curr_progress} %', end='\r')

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


def make_lstm_model(train_x, train_y, epochs=100, hps=None):

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
    print_train_progress_callback = CustomCallback(epochs)
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    lstm_model.fit(train_x, train_y, epochs=epochs, validation_split=0.25,  verbose=0, callbacks=[early_stopping_callback, print_train_progress_callback])

    return lstm_model


def forecast_lstm_model(model, test_x):
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


def experiment(stock_ticker, time_steps, drop_col=None, test_on_val=False, hps=None):
    """Function that creates and evaluates a single model.
    Returns the performance metrics of the created model.

    Args:
        stock_ticker (string): The target stock to be predicted.
        time_steps (int): The number of timesteps in the data window inputs.
        epochs (int): The maximum number of training epochs.

    Returns:
        dict: A dictionary of the performance metrics of the created model.
    """

    scaler, col_names, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=drop_col)

    if test_on_val:
        test_len = train_x.shape[0] * 25 // 100
        test_x = train_x[-test_len:]
        test_y = train_y[-test_len:]

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x, train_y, epochs=100, hps=hps)

    # get the model predictions
    predictions = forecast_lstm_model(lstm_model, test_x)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y = np.array([test_y[i, -1] for i in range(len(test_y))])

    # revert the normalization scalings done
    test_y = inverse_transform_data(test_y, scaler, col_names, feature="log_return")
    predictions = inverse_transform_data(predictions, scaler, col_names, feature="log_return")

    # get model performance statistics
    perf = get_lstm_model_perf(predictions, test_y)

    return perf, test_y, predictions



def feature_selection(stock_ticker, timesteps, repeats=20, hps=None):
    
    features = ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    num_features = len(features)
    dropped_features = features.copy()

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current: None)")
    

    model_perfs = []
    for _ in range(repeats):
        curr_model_perf, _, _ = experiment(stock_ticker, timesteps, drop_col=dropped_features, test_on_val=True, hps=hps)
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
            curr_model_perf, _, _ = experiment(stock_ticker, timesteps, drop_col=dropped_features, test_on_val=True, hps=hps)
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


def forward_feature_selection(stock_ticker, time_steps, repeats=10, hps=None):
    
    features = ['ad', 'wr', 'cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    num_features = len(features)
    dropped_features = features.copy()

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Round 0/{num_features}")
    print(f"Features Tested: 0/{num_features} (current features: [])")

    model_perfs = []

    for _ in range(repeats):
        curr_model_perf, _, _ = experiment(stock_ticker, time_steps, drop_col=dropped_features, test_on_val=True, hps=hps)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = mean(model_perfs)
    
    print(f"Current Mean Directional Accuracy: {round(curr_best_da, 6)}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")

    added_features = []
    curr_mean_model_perfs = [0] * (len(features) + 1)
    curr_mean_model_perfs[0] = curr_best_da

    for test_round in range(num_features):

        prev_round_best_da = curr_best_da

        for index, feature in enumerate(features):

            if feature in added_features:
                continue

            added_features.append(feature)

            print(f"Round {test_round + 1}/{num_features}")
            print(f"Features Tested: {index + 1}/{num_features} (current features: {added_features})")

            model_perfs = []

            dropped_features = features.copy()
            for added_feature in added_features:
                dropped_features.remove(added_feature)

            for _ in range(repeats):
                curr_model_perf, _, _ = experiment(stock_ticker, time_steps, drop_col=dropped_features, test_on_val=True, hps=hps)
                model_perfs.append(curr_model_perf['da'])

            curr_da = mean(model_perfs)
            curr_mean_model_perfs[index + 1] = curr_da

            if curr_da > curr_best_da:
                curr_best_da = curr_da

            added_features.remove(feature)

            print(f"Best Mean Directional Accuracy: {round(curr_best_da, 6)}")
            print(f"Current Mean Directional Accuracy: {round(curr_da, 6)}")
            print("===================================================")

        if curr_best_da <= prev_round_best_da:
            break

        curr_best_feature_index = curr_mean_model_perfs.index(curr_best_da) - 1
        added_features.append(features[curr_best_feature_index])

    dropped_features = features.copy()
    for added_feature in added_features:
        dropped_features.remove(added_feature)

    print(f"Best Mean Directional Accuracy: {round(curr_best_da, 6)}")
    print(f"Added Features: {added_features}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")
    
    return dropped_features


def get_hps(stock_ticker, dropped_features=None):

    # parameters of each model
    time_steps_list = [1, 5, 10, 15, 20]

    #['cmf', 'atr', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']

    if dropped_features is None:
        dropped_features = [None] * len(time_steps_list)

    hps_list = []

    for index, time_steps in enumerate(time_steps_list):
        _, _, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features[index])
        hps = get_optimal_hps(train_x, train_y)
        hps_list.append(hps)

    for i in hps_list:
        print(i)

    return hps_list



def main():
    # stock to be predicted
    stock_ticker = 'ALI'

    # parameters of each model
    time_steps = 1

    hps = {'units': 32, 'layers': 1, 'dropout': 0.2, 'tuner/epochs': 12, 'tuner/initial_epoch': 0, 'tuner/bracket': 2, 'tuner/round': 0}
    hps = None

    # how many models built (min = 2)
    repeats = 10

    # dropped features
    dropped_features = ['cmf', 'atr', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    ['cmf', 'atr', 'k_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate']

    ['cmf', 'atr', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    ['cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    ['wr', 'cmf', 'atr', 'adx', 'slope', 'k_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'psei_returns', 'sentiment']
    

    
    #bpi best ['wr', 'cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'gdp', 'real_interest_rate', 'roe', 'psei_returns']
    
    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _ = experiment(stock_ticker, time_steps, drop_col=dropped_features, hps=hps)
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



def batch_test(stock_ticker, dropped_features=None, hps_list=None):

    time_steps = [1, 5, 10, 15, 20]
    repeats = 10

    if dropped_features is None:
        dropped_features = [
            ['cmf', 'atr', 'cci', 'slope', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'], 
            ['wr', 'cmf', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'], 
            ['wr', 'cmf', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns'], 
            ['cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'sentiment'], 
            ['wr', 'atr', 'cci', 'adx', 'slope', 'k_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
        ]

    if hps_list is None:
        hps_list = [
            {'units': 256, 'layers': 3, 'dropout': 0.9, 'tuner/epochs': 34, 'tuner/initial_epoch': 12, 'tuner/bracket': 3, 'tuner/round': 2, 'tuner/trial_id': 'e720db3311a37430da6c2e24f4e6b43f'}, 
            {'units': 32, 'layers': 2, 'dropout': 0.7000000000000001, 'tuner/epochs': 34, 'tuner/initial_epoch': 12, 'tuner/bracket': 3, 'tuner/round': 2, 'tuner/trial_id': '88a24ad741bfa65308cfcbda576276cf'}, 
            {'units': 128, 'layers': 1, 'dropout': 0.6000000000000001, 'tuner/epochs': 12, 'tuner/initial_epoch': 4, 'tuner/bracket': 3, 'tuner/round': 1, 'tuner/trial_id': 'dea65f61f205ec5a19858684b99996d0'}, 
            {'units': 32, 'layers': 2, 'dropout': 0.4, 'tuner/epochs': 34, 'tuner/initial_epoch': 12, 'tuner/bracket': 3, 'tuner/round': 2, 'tuner/trial_id': '93f202b32b40ad88073a7d7a145ea8ad'},
            {'units': 64, 'layers': 1, 'dropout': 0.7000000000000001, 'tuner/epochs': 34, 'tuner/initial_epoch': 12, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': 'c8acd754bed6fd715c0e8fc4ca8b0e1b'}
        ]

    for index in range(len(time_steps)):
        print("===================================================")
        performances = []

        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")
            perf, _, _ = experiment(stock_ticker, time_steps[index], drop_col=dropped_features[index], hps=hps_list[index])
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


if __name__ == '__main__':

    warnings.simplefilter('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    # main()
    # visualize_returns('BPI')

    # hps = {'units': 128, 'layers': 1, 'dropout': 0.0, 'tuner/epochs': 34, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0}

    stock_ticker = 'AP'

    dropped_features = []
    time_steps = [1, 5, 10, 15, 20]

    for step in time_steps:
        curr_dropped_features = feature_selection(stock_ticker, step, repeats=15, hps=None)
        dropped_features.append(curr_dropped_features)

    hps_list = get_hps(stock_ticker, dropped_features)

    batch_test(stock_ticker, dropped_features, hps_list)
    print('===============')
    print(dropped_features)
    print(hps_list)


