from tensorflow import keras
from statistics import mean, stdev
import numpy as np
import pandas as pd
import os, sys, math
from sklearn.preprocessing import PowerTransformer, StandardScaler


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




def preprocess_data(data):
    """Function which proceses raw values into usable data
    (i.e. Closing Stock Prices -> Stock Returns)

    Args:
        data (pd.DataFrame): The raw data read from a csv file.

    Returns:
        pd.DataFrame: The processed dataset.
    """ 
    # get closing prices from data
    try:
        close = data['Adj Close']
    except KeyError:
        close = data['Close']

    # convert closing prices to stock returns
    stock_returns = []
    for i in range(1, len(data)):
        stock_return = math.log(close[i] / close[i - 1])
        # stock_return = ((close[i] - close[i - 1]) / close[i - 1]) * 100
        stock_returns.append(stock_return)


    volume = data['Volume']
    volumes = []
    for i in range(1, len(data)):
        volumes.append(volume[i] / 1000)

    # convert to dataframe
    processed_data = pd.DataFrame({
        'Stock Returns': stock_returns,
        'Volume': volumes
    })

    return processed_data




def train_test_split(data):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 8:2.

    Args:
        data (pd.DataFrame): The entire dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: The train and test datasets.
    """	
    test_len = len(data) * 2 // 10
    train, test = data[:-test_len], data[-test_len:]
    return train, test




def scale_data(train, test):
    """Applies Yeo-Johnson transformation to train and test datasets. 
    Each column or feature in the dataset is standardized separately.
    Args:
        train (pd.DataFrame): The test dataset.
        test (pd.DataFrame): The train dataset.
    Returns:
        Yeo-Johnson transformed train and test datasets.
    """
    # store column names
    col_names = list(train.columns)
    col_num = train.shape[1]

    # scale data for train & test data
    # scaler = StandardScaler()
    scaler = PowerTransformer()
    
    # Apply Yeo-Johnson Transform
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    # scale down outliers in train and test data
    for row in range(train.shape[0]):
        for col in range(col_num):
            if train[row, col] > 4.5:
                train[row, col] = 4.5
            elif train[row, col] < -4.5:
                train[row, col] = -4.5

    for row in range(test.shape[0]):
        for col in range(col_num):
            if test[row, col] > 4.5:
                test[row, col] = 4.5
            elif test[row, col] < -4.5:
                test[row, col] = -4.5

    # reconvert to dataframes
    train = pd.DataFrame({col: train[:, i] for i, col in enumerate(col_names)})
    test = pd.DataFrame({col: test[:, i] for i, col in enumerate(col_names)})

    return scaler, train, test, col_names




def invert_scaled_data(data, scaler, col_names, feature="Stock Returns"):
    """Inverses scaling done through Yeo-Johnson transformation. To be used
    with the predicted stock returns of the direction forecasting model.

    Args:
        data (np.array): The array representing scaled data.
        scaler (PowerTransformer): The scaler used to scale data.
        col_names (list): The list of the column names of the initaial dataset.
        feature (str, optional): The single feature to invert scaling. 
        Defaults to "Stock Returns".

    Returns:
        np.array: The array representing the unscaled data.
    """    
    unscaled_data = pd.DataFrame(np.zeros((len(data), len(col_names))), columns=col_names)
    unscaled_data[feature] = data
    unscaled_data = pd.DataFrame(scaler.inverse_transform(unscaled_data), columns=col_names)

    return unscaled_data[feature].values




def make_data_window(train, test, time_steps=1):
    """Creates data windows for the train and test datasets.
    Splits train and test datasets into train_x, train_y, test_x,
    and test_y datasets. The _x datasets represent model input datasets
    while _y datasets represent model target datasets. 

    i.e., for a sample dataset [[1, 2], [3, 4], [5, 6], [7, 8]]
    and a timestep of 2, the resulting _x dataset will be
    [[[1, 2], [3, 4]], [[3, 4], [5, 6]]] while the resulting _y
    dataset will be [5, 7].

    Args:
        train (pd.DataFrame): The train dataset.
        test (pd.DataFrame): The test dataset.
        time_steps (int, optional): How many time steps should
        be in each data window. Defaults to 1.

    Returns:
        pd.DataFrame (4): The train_x, train_y, test_x, and test_y datasets.
    """	
    # get the column index of stock returns in the dataframe
    stock_returns_index = train.columns.get_loc("Stock Returns")

    # convert dataframes into numpy arrays
    train = train.to_numpy()
    test = test.to_numpy()

    train_len = train.shape[0]
    test_len = test.shape[0]

    # x values: the input data window
    # y values: actual future values to be predicted from data window
    train_x, train_y, test_x, test_y = [], [], [], []

    for i in range(train_len):

        if (i + time_steps) < train_len:
            train_x.append([train[j, :] for j in range(i, i + time_steps)])
            train_y.append([train[j, stock_returns_index] for j in range(i + 1, i + time_steps + 1)])
            
    for i in range(test_len):
        
        if (i + time_steps) < test_len:
            test_x.append([test[j, :] for j in range(i, i + time_steps)])
            test_y.append([test[j, stock_returns_index] for j in range(i + 1, i + time_steps + 1)])


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y




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

    print("====================================================================")
    print(f"Total Ups: {perf['total_ups']}")
    print(f"Total Downs: {perf['total_downs']}")
    print("====================================================================")
    print(f"TP: {perf['tp']}")
    print(f"TN: {perf['tn']}")
    print(f"FP: {perf['fp']}")
    print(f"FN: {perf['fn']}")
    print("====================================================================")
    print(f"DA: {round(perf['da'], 6)}")
    print(f"UDA: {round(perf['uda'], 6)}")
    print(f"DDA: {round(perf['dda'], 6)}")
    print("====================================================================")




def experiment(stock_ticker, time_steps, epochs):
    """Function that creates and evaluates a single model.
    Returns the performance metrics of the created model.

    Args:
        stock_ticker (string): The target stock to be predicted.
        time_steps (int): The number of timesteps in the data window inputs.
        epochs (int): The maximum number of training epochs.

    Returns:
        dict: A dictionary of the performance metrics of the created model.
    """    	

    # get data from file
    raw_data = pd.read_csv(f'{stock_ticker}.csv')

    # preprocess data (i.e. calculate returns)
    data = preprocess_data(raw_data)

    # split and scale data
    train, test = train_test_split(data)
    scaler, train, test, col_names = scale_data(train, test)

    # get data slices or windows
    train_x, train_y, test_x, test_y = make_data_window(train, test, time_steps=time_steps)

    # create, compile, and fit an lstm model
    lstm_model = make_lstm_model(train_x, train_y, epochs=epochs)

    # get the model predictions
    predictions = forecast_lstm_model(lstm_model, test_x)

    # test_y has the shape of (samples, timesteps). Only the last timestep is the forecast target
    test_y = np.array([test_y[i, -1] for i in range(len(test_y))])

    # revert the normalization scalings done
    test_y = invert_scaled_data(test_y, scaler, col_names, feature="Stock Returns")
    predictions = invert_scaled_data(predictions, scaler, col_names, feature="Stock Returns")

    # get model performance statistics
    perf = get_lstm_model_perf(predictions, test_y)

    return perf




def main():
    os.chdir('data')

    # stock to be predicted
    stock_ticker = 'AP'

    # parameters of each model
    time_steps = 1
    epochs = 100

    # how many models built (min = 2)
    repeats = 2
    
    print("====================================================================")
    performances = []

    for i in range(repeats):
        print(f"Experiment {i + 1} / {repeats}")
        perf = experiment(stock_ticker, time_steps, epochs)
        performances.append(perf)
        print("====================================================================")

    mean_da = mean([perf['da'] for perf in performances])
    mean_uda = mean([perf['uda'] for perf in performances])
    mean_dda = mean([perf['dda'] for perf in performances])

    std_da = stdev([perf['da'] for perf in performances])
    std_uda = stdev([perf['uda'] for perf in performances])
    std_dda = stdev([perf['dda'] for perf in performances])
    
    optimistic_baseline = performances[0]['total_ups'] / (performances[0]['total_ups'] + performances[0]['total_downs'])
    pessimistic_baseline = 1 - optimistic_baseline
    
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