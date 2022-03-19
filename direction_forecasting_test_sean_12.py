# ANN implementation, random split with shuffling
from tensorflow import keras
from statistics import mean, stdev
import numpy as np
import pandas as pd
import os, sys, math, copy, random
from sklearn.preprocessing import PowerTransformer
from data_processing_test_sean import get_dataset, inverse_transform_data


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



def make_ann_model(train_x, train_y, epochs=5000):
    """Builds, compiles, fits, and returns an LSTM model based on
    provided training inputs and targets, as well as epochs and batch size.

    Args:
        train_x (np.array): The model inputs for training.
        train_y (np.array): The model target outputs for training.
        epochs (int, optional): Number of times the model is fitted. Defaults to 100.

    Returns:
        Model: The built Keras LSTM model.
    """	

    # The LSTM model to be used
    ann_model = keras.models.Sequential([
        keras.layers.Dense(units=train_x.shape[1], activation="linear"),
        keras.layers.Dense(units=30, activation="tanh"),
        keras.layers.Dense(units=1, activation="sigmoid")
    ])

    print_train_progress_callback = CustomCallback(epochs)

    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.7)

    ann_model.compile(loss='mean_squared_error', optimizer=optimizer)
    ann_model.fit(train_x, train_y, epochs=epochs, verbose=0, callbacks=[print_train_progress_callback])

    return ann_model


def forecast_ann_model(model, test_x):
    """Forecasts future values using a model and test input dataset.

    Args:
        model (Model): The built Keras model used for forecasting.
        test_x (np.array): The model inputs for testing and forecasting.

    Returns:
        np.array: A numpy array of the forecasted future values.
    """	
    predictions = []

    test_len = test_x.shape[0]
    test_timesteps = 1
    test_features = test_x.shape[1]

    # Make a separate prediction for each test data window in the test dataset
    for i in range(test_len):
        curr_progress = round(((i + 1) / test_len) * 100, 2)
        print(f'Prediction Progress: {curr_progress} %', end='\r')


        model_input = np.reshape(test_x[i], (1, test_x.shape[1]))
        prediction = model.predict(model_input)
        # the prediction has a shape of (1, timesteps, 1)
        # only need to get the last prediction value
        predictions.append(prediction[-1])
    print()

    predictions = (np.array(predictions)).flatten()
    return predictions


def get_ann_model_perf(predictions, actuals):
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
    total_ups = sum([1 if actuals[i] >= 0.5 else 0 for i in range(len(actuals))])
    total_downs = len(actuals) - total_ups

    # calculate true positives, true negatives, false positives, and false negatives
    tp = sum([1 if (predictions[i] >= 0.5 and actuals[i] >= 0.5) else 0 for i in range(predictions_len)])
    tn = sum([1 if (predictions[i] < 0.5 and actuals[i] < 0.5) else 0 for i in range(predictions_len)])
    fp = sum([1 if (predictions[i] >= 0.5 and actuals[i] < 0.5) else 0 for i in range(predictions_len)])
    fn = sum([1 if (predictions[i] < 0.5 and actuals[i] >= 0.5) else 0 for i in range(predictions_len)])

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


def experiment(scaler, col_names, train_x, train_y, test_x, test_y):

    train_x_copy = copy.deepcopy(train_x)
    train_y_copy = copy.deepcopy(train_y)
    test_x_copy = copy.deepcopy(test_x)
    test_y_copy = copy.deepcopy(test_y)

    # create, compile, and fit an lstm model
    ann_model = make_ann_model(train_x_copy, train_y_copy, epochs=5000)

    # get the model predictions
    predictions = forecast_ann_model(ann_model, test_x_copy)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y_copy = np.array(test_y_copy)

    # get model performance statistics
    perf = get_ann_model_perf(predictions, test_y_copy)

    print([1 if i >=0.5 else 0 for i in predictions[:10]])
    print(test_y_copy[:10])

    return perf, test_y_copy, predictions


def main():
    # stock to be predicted
    stock_ticker = 'AP'

    # parameters of each model
    time_steps = 1

    # how many models built (min = 2)
    repeats = 2

    # dropped features
    dropped_features = None
    # ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'p/e']

    scaler, col_names, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features)

    train_x = np.reshape(train_x, (train_x.shape[::2]))
    train_y = np.reshape(train_y, (train_y.shape[0]))
    test_x = np.reshape(test_x, (test_x.shape[::2]))
    test_y = np.reshape(test_y, (test_y.shape[0]))

    train_y = inverse_transform_data(train_y, scaler, col_names, feature="log_return")
    test_y = inverse_transform_data(test_y, scaler, col_names, feature="log_return")

    train_y = np.array([1 if i >= 0 else 0 for i in train_y])
    test_y = np.array([1 if i >= 0 else 0 for i in test_y])

    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _ = experiment(scaler, col_names, train_x, train_y, test_x, test_y)
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
    main()

    # pruned_features = feature_selection('AP', 5, repeats=2)
    # print(f"Dropped Features: {pruned_features}")