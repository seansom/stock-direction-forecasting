import pandas as pd
import numpy as np
import tensorflow as tf
import datetime, requests, json, math, shelve, sys, os, re, pathlib, nltk, shutil, zipfile

from decimal import Decimal
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from eventregistry import *
from statistics import mean
from pattern.en import lexeme

def requests_get(url):

    while True:
        try:
            return requests.get(url, timeout=10)
        except requests.exceptions.Timeout:
            print('Timeout. Restarting request...')
            continue


#get_technical_data START
def get_dates_one_year(testing=False):
    """Returns a 2-item tuple of dates in yyyy-mm-dd format 1 years in between today.

    Args:
        testing (bool, optional): If set to true, always returns ('2021-02-13', '2022-02-11'). Defaults to False.

    Returns:
        tuple: (from_date, to_date)
    """

    if testing:
        return ('2021-04-13', '2022-04-13')

    # generate datetime objects
    date_today = datetime.datetime.now()
    date_five_years_ago = date_today - datetime.timedelta(days=round(365.25 * 1))

    return (date_five_years_ago.strftime('%Y-%m-%d'), date_today.strftime('%Y-%m-%d'))


def get_trading_dates(stock_ticker, date_range, token):

    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={date_range[0]}&to={date_range[1]}"

    response = requests_get(url)
    data = response.json()

    # convert to pd.dataframe
    trading_dates = (pd.json_normalize(data))['date']

    return trading_dates


def get_technical_indicators(data):
    """Computes for log stock returns and technical indicators unavailable to EOD (e.g., A/D, CMF, WR).

    Args:
        data (pd.Dataframe): Dataframe containing dates and OHLCV stock data from EOD.

    Returns:
        pd.Dataframe: Dataframe containing dates, log stock returns and technical indicators.
    """    

    # get closing prices and volume from data
    try:
        close = data['adjusted_close']
    except KeyError:
        close = data['close']

    data_len = len(close)

    # compute log stock returns
    stock_returns = [np.NaN]
    for i in range(1, data_len):
        try:
            stock_return = 1 if math.log(Decimal((close[i])) / Decimal((close[i - 1]))) >= 0 else 0
        except TypeError:
            stock_return = 1 if math.log(Decimal(float(close[i])) / Decimal(float(close[i - 1]))) >= 0 else 0

        stock_returns.append(stock_return)

    # compute momentum values
    momentums = [np.NaN]
    for i in range(1, data_len):
        momentum = close[i] - close[i - 1]
        momentums.append(momentum)

    # compute A/D indicator values
    ad = []
    for i in range(data_len):
        try:
            ad_close = Decimal((close[i]))
            ad_low = Decimal((data['low'][i]))
            ad_high = Decimal((data['high'][i]))
        except TypeError:
            ad_close = Decimal(float(close[i]))
            ad_low = Decimal(float(data['low'][i]))
            ad_high = Decimal(float(data['high'][i]))

        if ad_low == ad_high:
            raise Exception(f'Error getting A/D indicator. A period has the same high and low price (zero division error).')
        
        mfm = ((ad_close - ad_low) - (ad_high - ad_close)) / (ad_high - ad_low)
        curr_ad =  mfm * data['volume'][i]
        ad.append(curr_ad)

    # compute William's %R indicator values
    wr_period = 14
    wr = [np.NaN] * (wr_period - 1)

    for i in range(wr_period, data_len + 1):
        wr_high = (data['high'][i - wr_period : i]).max()
        wr_low = (data['low'][i - wr_period : i]).min()
        wr_close = close[i - 1]

        if wr_low == wr_high:
            raise Exception(f"Error getting William's %R indicator. A period has the same highest and lowest price (zero division error).")
        
        try:
            curr_wr = Decimal((wr_high - wr_close)) / Decimal((wr_high - wr_low))
        except TypeError:
            curr_wr = Decimal(float(wr_high - wr_close)) / Decimal(float(wr_high - wr_low))

        wr.append(curr_wr)


    # convert to dataframe
    technical_indicators = pd.DataFrame({
        'date': data['date'],
        'stock_return': stock_returns,
        'momentum': momentums,
        'ad' : ad,
        'wr' : wr
    })

    return technical_indicators


def get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, date_range):
    """Gets daily technical indicator data from EOD API.

    Args:
        indicator (str): The indicator for use in EOD API calls. (e.g., rsi)
        period (str): The period used in computing the technical indicator.
        token (str): The EOD API key or token.
        stock_ticker (str): The stock ticker being examined (e.g., BPI).
        exchange (str): The stock exchange where the stock is being traded (e.g., PSE)
        date_range (tuple): A tuple of strings indicating the start and end dates for the requested data.

    Raises:
        Exception: Raises exception whenever gathered indicator data is insufficient to cover the specified
        date range. This may be fixed by increasing the timedelta used in computing the adjusted_first_day variable
        within the function.

    Returns:
        pd.Dataframe: Dataframe containing dates and specified technical indicator data
    """

    # compute an adjusted from or start date for the API
    first_trading_day_datetime = datetime.datetime.strptime(date_range[0],'%Y-%m-%d')
    adjusted_first_day = ((first_trading_day_datetime) - datetime.timedelta(days=100)).strftime('%Y-%m-%d')
    
    if indicator != 'macd':
        url = f"https://eodhistoricaldata.com/api/technical/{stock_ticker}.{exchange}?order=a&fmt=json&from={adjusted_first_day}&to={date_range[1]}&function={indicator}&period={period}&api_token={token}"
    else:
        url = f"https://eodhistoricaldata.com/api/technical/{stock_ticker}.{exchange}?order=a&fmt=json&from={adjusted_first_day}&to={date_range[1]}&function={indicator}&fast_period=4&slow_period=22&signal_period=3&api_token={token}"

    response = requests_get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    # remove rows with dates earlier than wanted from date
    for index, row in data.iterrows():

        # date of current row entry
        curr_date = datetime.datetime.strptime(row['date'],'%Y-%m-%d')

        # remove unneeded earlier row entries
        if curr_date < first_trading_day_datetime:
            data.drop(index, inplace=True)

        else:
            break

    # reset indices after dropping rows
    data = data.reset_index(drop=True)


    # raise an exception if the data from EOD API is insufficient to cover the date range specified
    # this error may be fixed by increasing timedelta used in computing adjusted_first_day
    if data['date'][0] != date_range[0]:
        raise Exception(f'Error getting {indicator} indicator for {stock_ticker}.')


    return data


def get_technical_data(stock_ticker, date_range):
    """Computes and gets technical dataset for a specific stock. To be used for model training.
    Args:
        stock_ticker ([type]): [description]
        date_range ([type]): [description]
    Raises:
        Exception: Raises exception whenever np.NaN is present in the processed technical dataset.
        This can occur if the EOD API data is missing data for a technical indicator on a specific day.
    Returns:
        pd.Dataframe: Dataframe representing the dates, log stock returns, and technical indicators.
    """

    # get API key/token from txt file
    with open('keys/EOD_API_key.txt') as file:
        token = file.readline()

    # get first and last trading days in the specified date range
    trading_days = get_trading_dates(stock_ticker, date_range, token)
    first_trading_day = trading_days[0]
    last_trading_day = trading_days.iat[-1]

    # adjust and add days to first trading day to be able to compute indicators with periods
    first_trading_day_datetime = datetime.datetime.strptime(first_trading_day,'%Y-%m-%d')
    adjusted_first_day = (first_trading_day_datetime - datetime.timedelta(days=40)).strftime('%Y-%m-%d')

    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={adjusted_first_day}&to={last_trading_day}"

    response = requests_get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    # # minimize volume data
    # data["volume"] = data["volume"].apply(lambda x: x)

    # compute returns and technical indicators not available on EOD
    technical_indicators = get_technical_indicators(data)
    data = data.merge(technical_indicators, on='date')

    # remove rows with dates earlier than wanted from date
    for index, row in data.iterrows():
        curr_date = datetime.datetime.strptime(row['date'],'%Y-%m-%d')

        if curr_date < first_trading_day_datetime:
            data.drop(index, inplace=True)

        else:
            break

    # reset indices after dropping rows
    data = data.reset_index(drop=True)

    # get available technical indicators from API. format: (indicator, period)
    EOD_indicators = [('sma', 10), ('wma', 10), ('stochastic', 14), ('rsi', 14), ('macd', 21), ('cci', 20)]


    for indicator, period in EOD_indicators:
        # print(indicator, period)
        indicator_data = get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, (first_trading_day, last_trading_day))
        data = data.merge(indicator_data, on='date')

    # remove unneeded features/columns in dataframe
    data = data.drop(columns=['open', 'high', 'low', 'adjusted_close', 'volume', 'close'])

    if data.isnull().values.any():
        raise Exception(f'Null value found in technical dataset for {stock_ticker}')

    return data
#get_technical_data END



#data_processing START
def scale_data(data):
    """Scales the data so that there won't be errors with the power transformer
    
    Args:
        data (pd.DataFrame): The entire dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: Scaled datasets.
    """
    
    linear_scaler = []

    for indicator in data.columns:
        if (data[f'{indicator}'] == 0).all():
            linear_scaler.append(1)
            continue

        scale_factor = round(math.log10(data[f'{indicator}'].abs().max()))
        linear_scaler.append(scale_factor)

        data[f'{indicator}'] = data[f'{indicator}'].apply(lambda x: float(Decimal(x) / Decimal(10 ** (scale_factor))))

    return data, linear_scaler


def train_test_split(data, time_steps):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 8:2.

    Args:
        data (pd.DataFrame): The entire dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: The train and test datasets.
    """	
    test_len = len(data) * 5 // 10
    train, test = data[:-test_len], data[-test_len - time_steps:]
    return train, test


def transform_data(train, test):
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
    scaler = MinMaxScaler()
    
    # Apply Yeo-Johnson Transform
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    # reconvert to dataframes
    train = pd.DataFrame({col: train[:, i] for i, col in enumerate(col_names)})
    test = pd.DataFrame({col: test[:, i] for i, col in enumerate(col_names)})

    return scaler, train, test, col_names


def inverse_transform_data(data, scaler, col_names, feature="stock_return"):
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


def data_processing(technical_data, drop_col=None, time_steps=1):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 8:2.

    Args:
        technical_data (pd.DataFrame): The dataset containing technical indicators.
        fundamental_data (pd.DataFrame): The dataset containing fundamental indicators.
        sentimental_data (pd.DataFrame): The dataset containing sentiment indicators.

    Returns:
        scaler, PowerTransformer: Scaler used in the power transform
        train, pd.DataFrame: The processed train dataset.
        test, pd.DataFrame: The processed test dataset.
        col_names, list: List of column names in the train and test datasets
    """	
    data = technical_data
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    #remove date column
    data.drop(columns=['date', 'macd', 'signal'], inplace=True)

    if drop_col is not None:
        data.drop(columns=drop_col, inplace=True)

    # raise exception if missing/null value is found in the combined dataset
    if data.isnull().values.any():
        raise Exception(f'Null value found in combined dataset.')
    
    #scale data
    scaled_data, linear_scaler = data, None

    #split data into train and test
    train, test = train_test_split(scaled_data, time_steps)
    
    #apply Yeo-Johnson Power Transfrom
    scaler, train, test, col_names = transform_data(train, test)

    return linear_scaler, scaler, train, test, col_names


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
    stock_returns_index = train.columns.get_loc("stock_return")

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
            # stock returns are not an input to the model, so take all features after the first
            train_x.append([train[j, 1:] for j in range(i, i + time_steps)])
            train_y.append([train[j, stock_returns_index] for j in range(i + 1, i + time_steps + 1)])
            
    for i in range(test_len):
        
        if (i + time_steps) < test_len:
            # stock returns are not an input to the model, so take all features after the first
            test_x.append([test[j, 1:] for j in range(i, i + time_steps)])
            test_y.append([test[j, stock_returns_index] for j in range(i + 1, i + time_steps + 1)])


    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y
#data_processing END


def get_dataset(stock_ticker, date_range=None, time_steps=1, drop_col=None):
    if date_range == None:
        date_range = get_dates_one_year(testing=True)

    os.chdir('data')

    # reset database if it gets too large (current max size = 20 MB)
    if os.path.exists('stock_database.dat') and os.path.getsize('stock_database.dat') > 20000000:
        os.remove('stock_database.bak')
        os.remove('stock_database.dat')
        os.remove('stock_database.dir')

    # open stock_database to see if data was already collected from API
    # otherwise, get data from API and store it in database
    stock_database = shelve.open('stock_database')
    stock_database_key = f"{stock_ticker} {date_range}"

    if stock_database_key in stock_database:
        technical_data = stock_database[stock_database_key]['technical_data']

    else:
        os.chdir('..')
        technical_data = get_technical_data(stock_ticker, date_range)
        os.chdir('data')

        stock_database[stock_database_key] = {
            'technical_data' : technical_data
        }
        
    stock_database.close()
    os.chdir('..')

    linear_scaler, scaler, train, test, col_names = data_processing(technical_data, drop_col, time_steps)

    train_x, train_y, test_x, test_y = make_data_window(train, test, time_steps)

    return linear_scaler, scaler, col_names, train_x, train_y, test_x, test_y


def main():
    stock_ticker = 'AP'

    linear_scaler, scaler, col_names, train_x, train_y, test_x, test_y = get_dataset(stock_ticker, date_range=get_dates_one_year(), time_steps=1, drop_col=None)    

    print(col_names)
    print(train_x.shape)
    


if __name__ == '__main__':
    main()
    # print(get_final_window('AP', date_range=None, time_steps=20).shape)