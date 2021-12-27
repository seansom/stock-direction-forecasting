# code base source: https://machinelearningmastery.com/update-lstm-networks-training-time-series-forecasting/
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
import numpy as np
import sys
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

import math
from sklearn.preprocessing import PowerTransformer

def log_preprocess_data(data):
    """Function which proceses raw values into usable data
    (i.e. Closing Stock Prices -> Stock Returns)
    Args:
        data (DataFrame): The raw data read from a csv file.
    Returns:
        DataFrame: The processed dataset.
    """
    # get closing prices from data
    try:
        close = data['Adj Close']
    except KeyError:
        close = data['Close']

    # convert closing prices to stock returns
    stock_returns = []
    for i in range(1, len(data)):
        stock_return = math.log((close[i] / close[i - 1]), 10)
        stock_returns.append(stock_return)

    # convert to dataframe
    processed_data = pd.DataFrame({
        'Stock Returns': stock_returns,
        'Volume': data['Volume'][1:]
    })

    return processed_data

def train_test_split(data):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 8:2.
    Args:
        data (DataFrame): The entire dataset.
    Returns:
        DataFrame, DataFrame: The train and test datasets.
    """

    test_len = len(data) * 2 // 10
    train, test = data[:-test_len], data[-test_len:]
    print(train)
    print(test)
    return train, test

def scale_data(train, test):
    """Applies standardization or Z-score normalization of the train
    and test datasets. Each column or feature in the dataset is
    standardized separately.
    Args:
        train (DataFrame): The test dataset.
        test (DataFrame): The train dataset.
    Returns:
        dict, DataFrame, DataFrame: The scaler which contains the
        means and standard deviations of each feature column, and the
        scaled train and test datasets.
    """


    # store column names
    col_names = list(train.columns)
    col_num = train.shape[1]

    # convert dataframes into numpy arrays
    train = train.to_numpy()
    test = test.to_numpy()

    #Apply Yeo-Johnson Transform
    pt = PowerTransformer()
    pt.fit(train)
    train = pt.transform(train)
    test = pt.transform(test)

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
    print(train)
    print(test)


    return train, test

def unscale_data(to_unscale, data):
    pt = PowerTransformer()
    col_names = list(to_unscale.columns)
    to_unscale = to_unscale.to_numpy()
    data = data.to_numpy()
    #get lambdas from original train data set
    pt.fit(data)
    to_unscale = pt.inverse_transform(to_unscale)
    to_unscale = pd.DataFrame({col: to_unscale[:, i] for i, col in enumerate(col_names)})
    print(to_unscale)
    return to_unscale

def run():
    # load dataset
    raw_data = pd.read_csv('BDOUY_4.csv')
    data = log_preprocess_data(raw_data)
    print(data)
    train, test = train_test_split(data)
    scaled_train, scaled_test = scale_data(train, test)
    unscale_data(scaled_train, train)
    unscale_data(scaled_test, train)



# entry point
run()
