from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import os, sys, math
from sklearn.preprocessing import PowerTransformer
from data_processing_test_sean_3 import get_dataset, inverse_transform_data


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



def make_lstm_model(train_x, train_y, epochs=100):
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
    lstm_model = keras.models.Sequential([
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.6),
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.6),
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.6),		
        keras.layers.Dense(units=1, activation="linear")
    ])

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


def experiment(stock_ticker, time_steps, epochs, drop_col=None):
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

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x, train_y, epochs=100)

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



def feature_selection(stock_ticker, timesteps):
    
    features = ['ad', 'wr', 'cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns']
    num_features = len(features)
    dropped_features = features.copy()

    repeats = 25

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current: None)")
    

    model_perfs = []
    for _ in range(repeats):
        curr_model_perf, _, _ = experiment(stock_ticker, timesteps, 100, drop_col=dropped_features)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = mean(model_perfs)
    
    print(f"Current Mean Directional Accuracy: {round(curr_best_da, 6)}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")


    for index, feature in enumerate(features):

        print(f"Features Tested: {index + 1}/{num_features} (current: {feature})")

        model_perfs = []
        dropped_features.remove(feature)

        for _ in range(repeats):
            curr_model_perf, _, _ = experiment(stock_ticker, timesteps, 100, drop_col=dropped_features)
            model_perfs.append(curr_model_perf['da'])

        curr_da = mean(model_perfs)

        print(f"Best Mean Directional Accuracy: {round(curr_best_da, 6)}")
        print(f"Current Mean Directional Accuracy: {round(curr_da, 6)}")

        if curr_da > curr_best_da:
            curr_best_da = curr_da
        else:
            dropped_features.append(feature)

        
        print(f"Dropped Features: {dropped_features}")
        print("===================================================")

    return dropped_features




def main():
    # stock to be predicted
    stock_ticker = 'AP'

    # parameters of each model
    time_steps = 1
    epochs = 100

    # how many models built (min = 2)
    repeats = 2

    # dropped features
    dropped_features = None#['ad', 'cmf', 'atr', 'cci', 'adx', 'signal', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'psei_returns', 'sentiment']

    
    #bpi best ['wr', 'cmf', 'atr', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'gdp', 'real_interest_rate', 'roe', 'psei_returns']
    
    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _ = experiment(stock_ticker, time_steps, epochs, drop_col=dropped_features)
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



def visualize_returns(stock_ticker):
    from matplotlib import pyplot
    os.chdir('data')

    perf, targets, predictions = experiment(stock_ticker, 1, 100)

    print("===================================================")
    print(perf)
    print("===================================================")

    pyplot.plot(targets)
    pyplot.plot(predictions)

    pyplot.title(f"{stock_ticker} Stock Returns")
    pyplot.legend(['Actual Stock Returns', 'Predicted Stock Returns'])

    pyplot.show()




if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    main()
    # visualize_returns('BPI')

    # pruned_features = feature_selection('BPI', 1)
    # print(f"Dropped Features: {pruned_features}")