from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import keras_tuner as kt
import os, sys, math, warnings, shutil
from data_processing_analysis_v3 import get_dataset, inverse_transform_data


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


def experiment(stock_ticker, time_steps, drop_col=None, test_on_val=False, hps=None, analysis_type='all'):
    """Function that creates and evaluates a single model.
    Returns the performance metrics of the created model.

    Args:
        stock_ticker (string): The target stock to be predicted.
        time_steps (int): The number of timesteps in the data window inputs.
        epochs (int): The maximum number of training epochs.

    Returns:
        dict: A dictionary of the performance metrics of the created model.
    """

    scaler, col_names, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=('2017-04-13', '2022-04-13'), time_steps=time_steps, drop_col=drop_col, analysis_type=analysis_type)

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


def get_FS(stock_ticker):
    if stock_ticker == 'ALI':
        return ['wr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'sentiment']
    elif stock_ticker == 'AP':
        return ['wr', 'cmf', 'rsi', 'cci', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    elif stock_ticker == 'BPI':
        return ['cmf', 'atr', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps']
    elif stock_ticker == 'JFC':
        return ['ad', 'wr', 'atr', 'rsi', 'cci', 'adx', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    elif stock_ticker == 'MER':
        return ['wr', 'cmf', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    elif stock_ticker == 'PGOLD':
        return ['wr', 'cmf', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    elif stock_ticker == 'SM':
        return ['atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'macd', 'signal', 'divergence', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    elif stock_ticker == 'TEL':
        return ['wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    else:
        raise Exception("Analysis is only for the eight stocks used in the research.")


def get_FS_HPS(stock_ticker):
    if stock_ticker == 'ALI':
        return {
            'time_step': 1,
            'drop_col': ['wr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'sentiment'],
            'hps': {
                'units': 32,
                'layers': 1,
                'dropout': 0.7
            }
        }
    elif stock_ticker == 'AP':
        return {
            'time_step': 1,
            'drop_col': ['wr', 'cmf', 'rsi', 'cci', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'],
            'hps': {
                'units': 256,
                'layers': 5,
                'dropout': 0
            }
        }
    elif stock_ticker == 'BPI':
        return {
            'time_step': 1,
            'drop_col': ['cmf', 'atr', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'real_interest_rate', 'roe', 'eps'],
            'hps': {
                'units': 64,
                'layers': 2,
                'dropout': 0.7
            }
        }
    elif stock_ticker == 'JFC':
        return {
            'time_step': 20,
            'drop_col': ['ad', 'wr', 'cmf', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'p/e'],
            'hps': {
                'units': 256,
                'layers': 1,
                'dropout': 0.6
            }
        }
    elif stock_ticker == 'MER':
        return {
            'time_step': 15,
            'drop_col': ['cmf', 'atr', 'rsi', 'adx', 'k_values', 'd_values', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns'],
            'hps': {
                'units': 256,
                'layers': 1,
                'dropout': 0.7
            }
        }
    elif stock_ticker == 'PGOLD':
        return {
            'time_step': 20,
            'drop_col': ['ad', 'wr', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'],
            'hps': {
                'units': 32,
                'layers': 2,
                'dropout': 0.4
            }
        }
    elif stock_ticker == 'SM':
        return {
            'time_step': 15,
            'drop_col': ['cmf', 'atr', 'rsi', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'],
            'hps': {
                'units': 64,
                'layers': 1,
                'dropout': 0.7
            }
        }
    elif stock_ticker == 'TEL':
        return {
            'time_step': 1,
            'drop_col': ['wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment'],
            'hps': {
                'units': 32,
                'layers': 3,
                'dropout': 0.6
            }
        }
    else:
        raise Exception("Analysis is only for the eight stocks used in the research.")


def analysis_FS(analysis_type):
    stock_ticker_list = ['ALI', 'AP', 'BPI', 'JFC', 'MER', 'PGOLD', 'SM', 'TEL']

    #stock_ticker_list = ['ALI']

    repeats = 10

    stock = []
    time_steps = []
    mean_DA = []
    mean_UDA = []
    mean_DDA = []
    std_dev_DA = []
    std_dev_UDA = []
    std_dev_DDA = []
    optimistic = []
    pessimistic = []

    for ticker in stock_ticker_list:
        print("===================================================")
        #fs_hps = get_FS_HPS(ticker)
        #time_step = fs_hps['time_step']
        time_step = 1
        #drop_col = fs_hps['drop_col']
        drop_col = get_FS(ticker)
        print(f"Stock: {ticker}; ; Time step: {time_step}; Analysis type: {analysis_type}")
        performances = []

        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")
            perf, _, _ = experiment(ticker, time_step, drop_col)
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
        stock.append(ticker)
        time_steps.append(time_step)
        mean_DA.append(mean_da)
        mean_UDA.append(mean_uda)
        mean_DDA.append(mean_dda)
        std_dev_DA.append(std_da)
        std_dev_UDA.append(std_uda)
        std_dev_DDA.append(std_dda)
        optimistic.append(optimistic_baseline)
        pessimistic.append(pessimistic_baseline)
    
    result = {
        'Stock': stock,
        'Time Steps': time_steps,
        'Mean DA': mean_DA,
        'Mean UDA': mean_UDA,
        'Mean DDA': mean_DDA,
        'Std Dev DA': std_dev_DA,
        'Std Dev UDA': std_dev_UDA,
        'Std Dev DDA': std_dev_DDA,
        'Optimistic': optimistic,
        'Pessimistic': pessimistic
    }

    result = pd.DataFrame(result)
    print(result)
    result.to_csv(f"{analysis_type}_analysis.csv")


def analysis_FS_HPS(analysis_type):
    stock_ticker_list = ['ALI', 'AP', 'BPI', 'JFC', 'MER', 'PGOLD', 'SM', 'TEL']

    #stock_ticker_list = ['ALI']

    repeats = 10

    stock = []
    time_steps = []
    mean_DA = []
    mean_UDA = []
    mean_DDA = []
    std_dev_DA = []
    std_dev_UDA = []
    std_dev_DDA = []
    optimistic = []
    pessimistic = []

    for ticker in stock_ticker_list:
        print("===================================================")
        fs_hps = get_FS_HPS(ticker)
        time_step = fs_hps['time_step']
        drop_col = fs_hps['drop_col']
        hps = fs_hps['hps']
        print(f"Stock: {ticker}; ; Time step: {time_step}; Analysis type: {analysis_type}")
        performances = []

        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")
            perf, _, _ = experiment(ticker, time_step, drop_col, hps=hps)
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
        stock.append(ticker)
        time_steps.append(time_step)
        mean_DA.append(mean_da)
        mean_UDA.append(mean_uda)
        mean_DDA.append(mean_dda)
        std_dev_DA.append(std_da)
        std_dev_UDA.append(std_uda)
        std_dev_DDA.append(std_dda)
        optimistic.append(optimistic_baseline)
        pessimistic.append(pessimistic_baseline)
    
    result = {
        'Stock': stock,
        'Time Steps': time_steps,
        'Mean DA': mean_DA,
        'Mean UDA': mean_UDA,
        'Mean DDA': mean_DDA,
        'Std Dev DA': std_dev_DA,
        'Std Dev UDA': std_dev_UDA,
        'Std Dev DDA': std_dev_DDA,
        'Optimistic': optimistic,
        'Pessimistic': pessimistic
    }

    result = pd.DataFrame(result)
    print(result)
    result.to_csv(f"{analysis_type}_analysis.csv")


def analysis(analysis_type='all'):
    if analysis_type == 'FS':
        analysis_FS(analysis_type)
        return
    elif analysis_type == 'FS_HPS':
        analysis_FS_HPS(analysis_type)
        return

    stock_ticker_list = ['ALI', 'AP', 'BPI', 'JFC', 'MER', 'PGOLD', 'SM', 'TEL']

    #stock_ticker_list = ['ALI']

    time_steps_list = [1, 5, 10, 15, 20]

    repeats = 10

    stock = []
    time_steps = []
    mean_DA = []
    mean_UDA = []
    mean_DDA = []
    std_dev_DA = []
    std_dev_UDA = []
    std_dev_DDA = []
    optimistic = []
    pessimistic = []

    for ticker in stock_ticker_list:

        for index in range(len(time_steps_list)):
            print("===================================================")
            print(f"Stock: {ticker}; Time step: {time_steps_list[index]}; Analysis type: {analysis_type}")
            performances = []

            for i in range(repeats):
                print(f"Experiment {i + 1} / {repeats}")
                perf, _, _ = experiment(ticker, time_steps_list[index], analysis_type=analysis_type)
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
            stock.append(ticker)
            time_steps.append(time_steps_list[index])
            mean_DA.append(mean_da)
            mean_UDA.append(mean_uda)
            mean_DDA.append(mean_dda)
            std_dev_DA.append(std_da)
            std_dev_UDA.append(std_uda)
            std_dev_DDA.append(std_dda)
            optimistic.append(optimistic_baseline)
            pessimistic.append(pessimistic_baseline)
    
    result = {
        'Stock': stock,
        'Time Steps': time_steps,
        'Mean DA': mean_DA,
        'Mean UDA': mean_UDA,
        'Mean DDA': mean_DDA,
        'Std Dev DA': std_dev_DA,
        'Std Dev UDA': std_dev_UDA,
        'Std Dev DDA': std_dev_DDA,
        'Optimistic': optimistic,
        'Pessimistic': pessimistic
    }

    result = pd.DataFrame(result)
    print(result)
    result.to_csv(f"{analysis_type}_analysis.csv")


def main():
    #analysis()
    #analysis('tech')
    #analysis('fund')
    #analysis('sent')
    analysis('FS')
    #analysis('FS_HPS')
    

if __name__ == '__main__':

    warnings.simplefilter('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    main()

