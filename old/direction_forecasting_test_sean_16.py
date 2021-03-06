# selective sampling split with train set shuffling
from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import os, sys, math, copy, random
from sklearn.preprocessing import PowerTransformer
from data_processing_test_sean_11 import get_dataset, inverse_transform_data


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
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.2),
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.2),
        keras.layers.LSTM(units=64, input_shape=train_x.shape[1:], return_sequences=True, recurrent_dropout=0.2),	

        keras.layers.Dense(units=16, activation="linear"),
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


def experiment(scaler, col_names, train_x, train_y, test_x, test_y):

    train_x_copy = copy.deepcopy(train_x)
    train_y_copy = copy.deepcopy(train_y)
    test_x_copy = copy.deepcopy(test_x)
    test_y_copy = copy.deepcopy(test_y)

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x_copy, train_y_copy, epochs=100)

    # get the model predictions
    predictions = forecast_lstm_model(lstm_model, test_x_copy)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y_copy = np.array([test_y_copy[i, -1] for i in range(len(test_y_copy))])

    # revert the normalization scalings done
    test_y_copy = inverse_transform_data(test_y_copy, scaler, col_names, feature="log_return")
    predictions = inverse_transform_data(predictions, scaler, col_names, feature="log_return")

    # get model performance statistics
    perf = get_lstm_model_perf(predictions, test_y_copy)

    # print(predictions[:10])
    # print(test_y_copy[:10])

    return perf, test_y_copy, predictions




def forward_feature_selection(stock_ticker, time_steps, repeats=25):
    
    _, col_names, correlations, _, _, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=None)

    features = col_names

    stock_returns_index = features.index('log_return')
    features.remove('log_return')
    correlations.pop(stock_returns_index)

    features = [x for _, x in sorted(zip(correlations, features), reverse=True)]

    num_features = len(features)
    dropped_features = features.copy()

    scaler, col_names, correlations, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features)

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current features: [])")
    
    model_perfs = []

    validation_len = train_x.shape[0] * 2 // 10

    # adjusted_train_indices = list(range(train_x.shape[0] - validation_len))

    adjusted_train_x = train_x[:-validation_len]
    adjusted_train_y = train_y[:-validation_len]
    validation_x = train_x[-validation_len:]
    validation_y = train_y[-validation_len:]

    shuffled_train_indices = list(range(adjusted_train_x.shape[0]))

    random.seed(0)
    random.shuffle(shuffled_train_indices)

    shuffled_train_x = np.array([adjusted_train_x[i] for i in shuffled_train_indices])
    shuffled_train_y = np.array([adjusted_train_y[i] for i in shuffled_train_indices])

    for i in range(repeats):
        print(f"Experiment {i + 1}/{repeats}")
        curr_model_perf, _, _ = experiment(scaler, col_names, shuffled_train_x, shuffled_train_y, validation_x, validation_y)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = round(mean(model_perfs), 6)
    
    print(f"Best Mean Directional Accuracy: {curr_best_da}")
    print(f"Current Mean Directional Accuracy: {curr_best_da}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")


    for index, feature in enumerate(features):

        dropped_features.remove(feature)

        print(f"Features Tested: {index + 1}/{num_features} (current features: {[feature for feature in features if feature not in dropped_features]})")

        model_perfs = []

        scaler, col_names, correlations, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features)

        adjusted_train_x = train_x[:-validation_len]
        adjusted_train_y = train_y[:-validation_len]
        validation_x = train_x[-validation_len:]
        validation_y = train_y[-validation_len:]

        shuffled_train_x = np.array([adjusted_train_x[i] for i in shuffled_train_indices])
        shuffled_train_y = np.array([adjusted_train_y[i] for i in shuffled_train_indices])

        for i in range(repeats):
            print(f"Experiment {i + 1}/{repeats}")
            curr_model_perf, _, _ = experiment(scaler, col_names, shuffled_train_x, shuffled_train_y, validation_x, validation_y)
            model_perfs.append(curr_model_perf['da'])

        curr_mean_da = round(mean(model_perfs), 6)

        if curr_mean_da >= curr_best_da:
            curr_best_da = curr_mean_da
        else:
            dropped_features.append(feature)

        print(f"Best Mean Directional Accuracy: {curr_best_da}")
        print(f"Current Mean Directional Accuracy: {curr_mean_da}")
        print(f"Dropped Features: {dropped_features}")
        print("===================================================")

    added_features = [feature for feature in features if feature not in dropped_features]

    print(f"Best Mean Directional Accuracy: {curr_best_da}")
    print(f"Selected Features: {added_features}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")

    return dropped_features


def backward_feature_selection(stock_ticker, time_steps, repeats=25):
    
    scaler, col_names, correlations, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=None)

    features = col_names.copy()

    stock_returns_index = features.index('log_return')
    features.remove('log_return')
    correlations.pop(stock_returns_index)

    features = [x for _, x in sorted(zip(correlations, features))]

    num_features = len(features)
    dropped_features = []

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current dropped features: [])")
    
    model_perfs = []

    validation_len = train_x.shape[0] * 2 // 10

    # adjusted_train_indices = list(range(train_x.shape[0] - validation_len))

    adjusted_train_x = train_x[:-validation_len]
    adjusted_train_y = train_y[:-validation_len]
    validation_x = train_x[-validation_len:]
    validation_y = train_y[-validation_len:]

    shuffled_train_indices = list(range(adjusted_train_x.shape[0]))

    random.seed(0)
    random.shuffle(shuffled_train_indices)

    shuffled_train_x = np.array([adjusted_train_x[i] for i in shuffled_train_indices])
    shuffled_train_y = np.array([adjusted_train_y[i] for i in shuffled_train_indices])

    for i in range(repeats):
        print(f"Experiment {i + 1}/{repeats}")
        curr_model_perf, _, _ = experiment(scaler, col_names, shuffled_train_x, shuffled_train_y, validation_x, validation_y)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = round(mean(model_perfs), 6)
    
    print(f"Best Mean Directional Accuracy: {curr_best_da}")
    print(f"Current Mean Directional Accuracy: {curr_best_da}")
    print(f"Current Features: {features}")
    print("===================================================")


    for index, feature in enumerate(features):

        dropped_features.append(feature)

        print(f"Features Tested: {index + 1}/{num_features} (current dropped features: {dropped_features})")

        model_perfs = []

        scaler, col_names, correlations, train_x, train_y, _, _ = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features)

        adjusted_train_x = train_x[:-validation_len]
        adjusted_train_y = train_y[:-validation_len]
        validation_x = train_x[-validation_len:]
        validation_y = train_y[-validation_len:]

        shuffled_train_x = np.array([adjusted_train_x[i] for i in shuffled_train_indices])
        shuffled_train_y = np.array([adjusted_train_y[i] for i in shuffled_train_indices])

        for i in range(repeats):
            print(f"Experiment {i + 1}/{repeats}")
            curr_model_perf, _, _ = experiment(scaler, col_names, shuffled_train_x, shuffled_train_y, validation_x, validation_y)
            model_perfs.append(curr_model_perf['da'])

        curr_mean_da = round(mean(model_perfs), 6)

        if curr_mean_da >= curr_best_da:
            curr_best_da = curr_mean_da
        else:
            dropped_features.remove(feature)

        print(f"Best Mean Directional Accuracy: {curr_best_da}")
        print(f"Current Mean Directional Accuracy: {curr_mean_da}")
        print(f"Current Features: {[feature for feature in features if feature not in dropped_features]}")
        print("===================================================")

    added_features = [feature for feature in features if feature not in dropped_features]

    print(f"Best Mean Directional Accuracy: {curr_best_da}")
    print(f"Selected Features: {added_features}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")

    return dropped_features

def main():
    # stock to be predicted
    stock_ticker = 'PGOLD'

    # parameters of each model
    time_steps = 20

    # how many models built (min = 2)
    repeats = 2

    # dropped features
    dropped_features = None
    ['volatility5', 'slope14', 'divergence', 'wr14', 'p/e', 'd_values_x', 'rsi14', 'macd26', 'ad', 'intraday_return', 'cci20', 'atr5', 'signal', 'cci5', 'slope3', 'adx5', 'adx14', 'sentiment', 'real_interest_rate', 'roe', 'gdp', 'inflation', 'atr14', 'psei_returns', 'eps']
    # AP forward ['intraday_return', 'wr5', 'slope2', 'rsi5', 'slope5', 'rsi14', 'cmf5', 'cci5', 'divergence', 'cci20', 'slope14', 'mband', 'k_values_y', 'k_values_x', 'lband', 'volatility5', 'signal', 'eps', 'atr5', 'd_values_y', 'inflation', 'roe', 'atr14', 'gdp', 'psei_returns', 'real_interest_rate', 'sentiment']
    # bpi backward ['macd26', 'eps', 'd_values_x', 'signal', 'cmf20', 'adx5', 'divergence', 'slope2', 'slope5', 'wr5']
    # bpi forward ['slope4', 'cci5', 'slope5', 'rsi5', 'p/e', 'atr14', 'atr5', 'slope2', 'lband', 'mband', 'uband', 'sentiment', 'rsi14', 'divergence', 'adx14', 'cmf5', 'k_values_y', 'k_values_x', 'cci20', 'cmf20', 'inflation', 'ad', 'gdp', 'signal', 'd_values_x', 'slope14', 'real_interest_rate', 'roe', 'eps']
    #['volatility5', 'wr14', 'mband', 'k_values_y', 'k_values_x', 'uband', 'd_values_y', 'd_values_x', 'rsi14', 'macd26', 'slope2', 'slope4', 'cmf20', 'atr5', 'signal', 'cci5', 'slope3', 'adx5', 'adx14', 'sentiment', 'real_interest_rate', 'roe', 'gdp', 'inflation', 'atr14', 'psei_returns']
    # ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'p/e']

    scaler, col_names, correlations, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, drop_col=dropped_features)

    # print(col_names)
    # sys.exit()

    shuffled_train_indices = list(range(train_x.shape[0]))
    random.seed(0)
    random.shuffle(shuffled_train_indices)

    shuffled_train_x = np.array([train_x[i] for i in shuffled_train_indices])
    shuffled_train_y = np.array([train_y[i] for i in shuffled_train_indices])

    
    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _ = experiment(scaler, col_names, shuffled_train_x, shuffled_train_y, test_x, test_y)
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    compat.v1.logging.set_verbosity(compat.v1.logging.ERROR)

    main()

    # pruned_features = forward_feature_selection('ALI', 1, repeats=10)
    # print(f"Dropped Features: {pruned_features}")