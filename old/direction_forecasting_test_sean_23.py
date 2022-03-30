# no random splitting with shuffling, walk forward validation with SHITTON of indicators, delayed prediction
from tensorflow import keras, compat
from statistics import mean, stdev
import numpy as np
import pandas as pd
import os, sys, math, copy, random
from sklearn.preprocessing import PowerTransformer
from data_processing_test_sean_13 import get_dataset, inverse_transform_data


class CustomCallback(keras.callbacks.Callback):
    """A callback class used to print the progress of model fitting
    after each epoch.
    """	
    def __init__(self, epochs):
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        curr_progress = round(((epoch + 1) / self.epochs) * 100, 2)
        print(f'Training Progress: {curr_progress} % (val_binary_accuracy: {round(logs["val_binary_accuracy"], 6)})', end='\r')

    def on_train_end(self, logs=None):
        print()



def make_lstm_model(input_shape):

    lstm_model = keras.models.Sequential([
        keras.layers.LSTM(units=64, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2),
        keras.layers.LSTM(units=64, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2),
        keras.layers.LSTM(units=64, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2),

        keras.layers.Dense(units=16, activation='sigmoid'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

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


def experiment(data):

    data_copy = copy.deepcopy(data)
    data_len = len(data)
    input_shape = data_copy[0]['train_x'].shape[1:]

    lstm_model = make_lstm_model(input_shape)
    lstm_weights = lstm_model.get_weights()

    predictions = []
    actuals = []

    epochs = 100
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=16, mode='min', restore_best_weights=True)
    print_train_progress_callback = CustomCallback(epochs)

    curr_directional_accuracy = 0

    for index in range(data_len):

        print(f'Round {index + 1}/{data_len}')

        train_x = data_copy[index]['train_x']
        train_y = data_copy[index]['train_y']
        test_x = data_copy[index]['test_x']
        test_y = data_copy[index]['test_y']

        shuffled_train_indices = list(range(train_x.shape[0]))

        random.seed(0)
        random.shuffle(shuffled_train_indices)

        shuffled_train_x = np.array([train_x[i] for i in shuffled_train_indices])
        shuffled_train_y = np.array([train_y[i] for i in shuffled_train_indices])

        # if curr_directional_accuracy < 0.6:
        lstm_model.set_weights(lstm_weights)
        lstm_model.reset_states()

        lstm_model.fit(shuffled_train_x, shuffled_train_y, epochs=512, validation_split=0.25, verbose=0, callbacks=[early_stopping_callback, print_train_progress_callback])

        curr_predictions = list(forecast_lstm_model(lstm_model, test_x))
        curr_actuals = [test_y[i, -1] for i in range(len(test_y))]

        predictions = predictions + curr_predictions
        actuals = actuals + curr_actuals

        correct_predictions_num = sum([1 if (curr_predictions[i] >= 0.5 and curr_actuals[i] >= 0.5) or (curr_predictions[i] < 0.5 and curr_actuals[i] < 0.5) else 0 for i in range(len(curr_predictions))])
        curr_directional_accuracy = correct_predictions_num / len(curr_predictions)
        print(f"Batch DA: {round(curr_directional_accuracy, 6)}")


    # get model performance statistics
    perf = get_lstm_model_perf(predictions, actuals)


    return perf, actuals, predictions



def feature_selection(stock_ticker, time_steps, train_size, test_size, repeats=5):
    
    features = ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']
    num_features = len(features)
    dropped_features = []

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Feature Round 0/{num_features}")
    print(f"Features Tested: 0/{num_features} (current dropped features: [])")
    
    model_perfs = []

    data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=dropped_features)

    for index in range(len(data)):
        data[index]['test_x'] = data[index]['train_x'][-test_size:]
        data[index]['test_y'] = data[index]['train_y'][-test_size:]
        data[index]['train_x'] = data[index]['train_x'][:-test_size]
        data[index]['train_y'] = data[index]['train_y'][:-test_size]

    for i in range(repeats):
        print(f"Experiment {i + 1}/{repeats}")
        curr_model_perf, _, _ = experiment(data)
        model_perfs.append(curr_model_perf['da'])

    curr_best_da = round(mean(model_perfs), 6)
    
    print(f"Current Mean Directional Accuracy: {round(curr_best_da, 6)}")
    print(f"Dropped Features: {dropped_features}")
    print("===================================================")

    curr_mean_model_perfs = [0] * (len(features) + 1)
    curr_mean_model_perfs[0] = curr_best_da

    for test_round in range(num_features):

        prev_round_best_da = curr_best_da

        for index, feature in enumerate(features):

            if feature in dropped_features:
                continue

            dropped_features.append(feature)

            print(f"Feature Round {test_round + 1}/{num_features}")
            print(f"Features Tested: {index + 1}/{num_features} (current dropped features: {dropped_features})")

            model_perfs = []
            
            data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=dropped_features)

            for i in range(len(data)):
                data[i]['test_x'] = data[i]['train_x'][-test_size:]
                data[i]['test_y'] = data[i]['train_y'][-test_size:]
                data[i]['train_x'] = data[i]['train_x'][:-test_size]
                data[i]['train_y'] = data[i]['train_y'][:-test_size]

            for i in range(repeats):
                print(f"Experiment {i + 1}/{repeats}")
                curr_model_perf, _, _ = experiment(data)
                model_perfs.append(curr_model_perf['da'])

            curr_da = round(mean(model_perfs), 6)
            curr_mean_model_perfs[index + 1] = curr_da

            if curr_da > curr_best_da:
                curr_best_da = curr_da

            dropped_features.remove(feature)

            print(f"Best Mean Directional Accuracy: {round(curr_best_da, 6)}")
            print(f"Current Mean Directional Accuracy: {round(curr_da, 6)}")
            print("===================================================")

        if curr_best_da <= prev_round_best_da:
            break

        curr_best_feature_index = curr_mean_model_perfs.index(curr_best_da) - 1
        dropped_features.append(features[curr_best_feature_index])

    return dropped_features



def simple_feature_selection(stock_ticker, time_steps, train_size, test_size, repeats=25):
    
    data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=None)

    features = data[0]['col_names'].copy()
    correlations = data[0]['correlations'].copy()

    stock_returns_index = features.index('log_return')
    features.remove('log_return')
    correlations.pop(stock_returns_index)

    features = [x for _, x in sorted(zip(correlations, features), reverse=True)]

    num_features = len(features)
    dropped_features = features.copy()

    data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=dropped_features)

    print("===================================================")
    print("Starting Feature Selection...")
    print(f"Features Tested: 0/{num_features} (current features: [])")
    
    model_perfs = []

    for index in range(len(data)):
        data[index]['test_x'] = data[index]['train_x'][-test_size:]
        data[index]['test_y'] = data[index]['train_y'][-test_size:]
        data[index]['train_x'] = data[index]['train_x'][:-test_size]
        data[index]['train_y'] = data[index]['train_y'][:-test_size]

    for i in range(repeats):
        print(f"Experiment {i + 1}/{repeats}")
        curr_model_perf, _, _ = experiment(data)
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

        data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=dropped_features)

        for index in range(len(data)):
            data[index]['test_x'] = data[index]['train_x'][-test_size:]
            data[index]['test_y'] = data[index]['train_y'][-test_size:]
            data[index]['train_x'] = data[index]['train_x'][:-test_size]
            data[index]['train_y'] = data[index]['train_y'][:-test_size]

        for i in range(repeats):
            print(f"Experiment {i + 1}/{repeats}")
            curr_model_perf, _, _ = experiment(data)
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




def main():
    # stock to be predicted
    stock_ticker = 'AP'

    # parameters of each model
    time_steps = 20
    train_size = 1004
    test_size = 21

    # how many models built (min = 2)
    repeats = 2

    # dropped features
    dropped_features = None
    # ALI ['lband', 'p/e', 'mband', 'uband', 'wr5', 'k_values_y', 'k_values_x', 'd_values_y', 'd_values_x', 'macd26', 'rsi14', 'rsi5', 'cmf20', 'atr5', 'roe', 'signal', 'slope2', 'slope5', 'adx14', 'cci20', 'cci5', 'gdp', 'slope4', 'adx5', 'inflation', 'intraday_return', 'psei_returns', 'atr14', 'slope3', 'eps']
    ['divergence', 'lband', 'p/e', 'mband', 'wr5', 'uband', 'wr14', 'k_values_x', 'd_values_y', 'd_values_x', 'rsi5', 'macd26', 'rsi14', 'ad', 'atr5', 'cmf20', 'real_interest_rate', 'sentiment', 'slope2', 'slope5', 'signal', 'gdp', 'cci5', 'adx14', 'slope4', 'adx5', 'intraday_return', 'atr14', 'inflation', 'psei_returns', 'slope3', 'eps']

    # AP (1, 1004, 21) ['wr', 'rsi14', 'cci20', 'adx14', 'slope14', 'k_values_x', 'd_values_x', 'macd26', 'signal', 'divergence', 'slope2', 'slope5', 'volatility5', 'uband', 'mband', 'lband', 'atr5', 'rsi5', 'adx5', 'k_values_y', 'd_values_y', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'p/e', 'psei_returns', 'sentiment']
    # ALI (1, 1004, 21) ['volume', 'cmf', 'atr14', 'rsi14', 'cci20', 'adx14', 'slope14', 'k_values_x', 'macd26', 'signal', 'divergence', 'slope2', 'slope3', 'slope4', 'slope5', 'volatility5', 'uband', 'mband', 'lband', 'atr5', 'k_values_y', 'd_values_y', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'sentiment']
    # PGOLD (1, 1004, 21) ['ad', 'wr', 'cmf', 'atr14', 'rsi14', 'cci20', 'adx14', 'slope14', 'k_values_x', 'd_values_x', 'macd26', 'signal', 'divergence', 'slope2', 'slope3', 'slope4', 'slope5', 'volatility5', 'mband', 'lband', 'rsi5', 'cci5', 'adx5', 'k_values_y', 'd_values_y', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e']
    # BPI (1, 1004, 21) ['cmf20', 'cmf5', 'atr14', 'adx14', 'k_values_x', 'signal', 'divergence', 'slope2', 'slope4', 'slope5', 'volatility5', 'uband', 'mband', 'lband', 'atr5', 'rsi5', 'cci5', 'adx5', 'k_values_y', 'd_values_y', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']

    ['cmf5', 'volume', 'lband', 'p/e', 'close', 'adjusted_close', 'mband', 'open', 'high', 'low', 'uband', 'k_values_y', 'k_values_x', 'd_values_y', 'd_values_x', 'macd26', 'rsi14', 'rsi5', 'cmf20', 'atr5', 'roe', 'real_interest_rate', 'signal', 'slope2', 'sentiment', 'slope5', 'adx14', 'cci20', 'cci5', 'gdp', 'adx5', 'inflation', 'psei_returns', 'atr14', 'slope3', 'eps']

    # ALL ['ad', 'wr', 'cmf', 'atr', 'rsi', 'cci', 'adx', 'slope', 'k_values', 'd_values', 'macd', 'signal', 'divergence', 'gdp', 'inflation', 'real_interest_rate', 'roe', 'eps', 'p/e', 'psei_returns', 'sentiment']

    data = get_dataset(stock_ticker, date_range=None, time_steps=time_steps, train_size=train_size, test_size=test_size, drop_col=dropped_features)

    print("===================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf, _, _ = experiment(data)
        performances.append(perf)
        print_model_performance(perf)
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

    # pruned_features = simple_feature_selection('BPI', 20, 1004, 21, repeats=8)
    # print(f"Dropped Features: {pruned_features}")