import pandas as pd
import numpy as np
from decimal import Decimal
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
import datetime, requests, json, math, sys, os

#get_technical_data START
def get_dates_five_years(testing=False):
    """Returns a 2-item tuple of dates in yyyy-mm-dd format 5 years in between today.
    Args:
        testing (bool, optional): If set to true, always returns ('2017-02-13', '2022-02-11'). Defaults to False.
    Returns:
        tuple: (from_date, to_date)
    """

    if testing:
        return ('2017-02-13', '2022-02-11')

    # generate datetime objects
    date_today = datetime.datetime.now()
    date_five_years_ago = date_today - datetime.timedelta(days=round(365.25 * 5))

    return (date_five_years_ago.strftime('%Y-%m-%d'), date_today.strftime('%Y-%m-%d'))


def get_trading_dates(stock_ticker, date_range, token):
    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={date_range[0]}&to={date_range[1]}"

    response = requests.get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    return data['date']


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
        stock_return = math.log(Decimal(close[i]) / Decimal(close[i - 1]))
        stock_returns.append(stock_return)

    # compute A/D indicator values
    ad = []
    for i in range(data_len):
        ad_close = Decimal(close[i])
        ad_low = Decimal(data['low'][i])
        ad_high = Decimal(data['high'][i])

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
        
        curr_wr = Decimal(wr_high - wr_close) / Decimal(wr_high - wr_low)
        wr.append(curr_wr)

    # compute Chaulkin Money Flow indicator
    cmf_period = 20

    mfv = []
    cmf = [np.NaN] * (cmf_period - 1)

    for i in range(data_len):
        cmf_close = Decimal(close[i])
        cmf_low = Decimal(data['low'][i])
        cmf_high = Decimal(data['high'][i])
        cmf_volume = data['volume'][i]

        if cmf_low == cmf_high:
            raise Exception(f'Error getting CMF indicator. A period has the same high and low price (zero division error).')

        curr_mfv = (((cmf_close - cmf_low) - (cmf_high - cmf_close)) / (cmf_high - cmf_low)) *  cmf_volume
        mfv.append(curr_mfv)

    for i in range(cmf_period, data_len + 1):
        curr_cmf = sum(mfv[i - cmf_period : i]) / sum(data['volume'][i - cmf_period : i])
        cmf.append(curr_cmf)


    # convert to dataframe
    technical_indicators = pd.DataFrame({
        'date': data['date'],
        'log_return': stock_returns,
        'ad' : ad,
        'wr' : wr,
        'cmf' : cmf
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
    
    url = f"https://eodhistoricaldata.com/api/technical/{stock_ticker}.{exchange}?order=a&fmt=json&from={adjusted_first_day}&to={date_range[1]}&function={indicator}&period={period}&api_token={token}"

    response = requests.get(url)
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
    adjusted_first_day = (first_trading_day_datetime - datetime.timedelta(days=30)).strftime('%Y-%m-%d')

    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={adjusted_first_day}&to={last_trading_day}"

    response = requests.get(url)
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
    EOD_indicators = [('atr', 14), ('rsi', 14), ('cci', 20), ('adx', 14), ('slope', 14), ('stochastic', 14), ('macd', 26)]
    #[('atr', 14), ('rsi', 14), ('cci', 20), ('adx', 14), ('slope', 3), ('stochastic', 14), ('macd', 26)]
    #[('atr', 14), ('rsi', 14), ('cci', 20), ('adx', 14)]

    for indicator, period in EOD_indicators:
        indicator_data = get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, (first_trading_day, last_trading_day))
        data = data.merge(indicator_data, on='date')

    # remove unneeded features/columns in dataframe
    data = data.drop(columns=['open', 'high', 'low', 'adjusted_close', 'volume', 'close'])

    if data.isnull().values.any():
        raise Exception(f'Null value found in technical dataset for {stock_ticker}')

    # linearly scale up/down indicators to avoid power transform errors
    # for indicator in data.columns:
    #     if indicator != 'date':
    #         scale_factor = round(math.log10(data[f'{indicator}'].abs().max()))
    #         data[f'{indicator}'] = data[f'{indicator}'].apply(lambda x: float(Decimal(x) / Decimal(10 ** (scale_factor))))

    return data
#get_technical_data END

#get_fundamental_data START
def get_fundamental_trading_dates(stock_ticker, date_range, token):
    """compiles trading dates and adjusted closing prices to be used as a base data set for the fundamental data
    
    Args:
        stock_ticker (str): stock ticker of the company to get data from
        date_range (tuple): range of dates to extract data from
        token (str): eod api token
    
    Returns:
        pd.Dataframe: Dataframe containing the trading dates and adjusted closing prices
    """

    #get raw data from eod
    exchange = 'PSE'

    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={date_range[0]}&to={date_range[1]}"
    response = requests.get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    #filter out date and adjusted closing price
    try:
        close = 'adjusted_close'
    except KeyError:
        close = 'close'

    return data[['date', close]]


def get_psei_returns(date_range, token):
    """computes log stock returns for PSEI
    
    Args:
        date_range (tuple): range of dates to extract data from
        token (str): eod api token
   
    Returns:
        pd.Dataframe: Dataframe containing dates and log stock returns for PSEI
    """

    #get raw data from eod
    stock_ticker = 'PSEI'
    exchange = 'INDX'

    # compute an adjusted from or start date for the API
    first_trading_day_datetime = datetime.datetime.strptime(date_range[0],'%Y-%m-%d')
    adjusted_first_day = ((first_trading_day_datetime) - datetime.timedelta(days=100)).strftime('%Y-%m-%d')

    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={adjusted_first_day}&to={date_range[1]}"
    response = requests.get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    # compute log stock returns
    try:
        close = data['adjusted_close']
    except KeyError:
        close = data['close']

    data_len = len(close)

    stock_returns = [np.NaN]
    for i in range(1, data_len):
        stock_return = math.log(Decimal(close[i]) / Decimal(close[i - 1]))
        stock_returns.append(stock_return)

    #convert into pd.Dataframe    
    psei_returns = pd.DataFrame({
        'date': data['date'],
        'psei_returns': stock_returns
    })

    # remove rows with dates earlier than wanted from date
    for index, row in psei_returns.iterrows():

        # date of current row entry
        curr_date = datetime.datetime.strptime(row['date'],'%Y-%m-%d')

        # remove unneeded earlier row entries
        if curr_date < first_trading_day_datetime:
            psei_returns.drop(index, inplace=True)

        else:
            break

    # reset indices after dropping rows
    psei_returns = psei_returns.reset_index(drop=True)
    
    return psei_returns


def get_fundamental_indicator_from_EOD(stock_ticker, token):
    """get fundamental indicators from EOD
    Args:
        stock_ticker (str): stock ticker of the company to get data from
        token (str): eod api token
    Returns:
        json: data in json format containing fundamental indicators from EOD
    """

    #get raw data from eod
    exchange = 'PSE'

    url = f"https://eodhistoricaldata.com/api/fundamentals/{stock_ticker}.{exchange}?api_token={token}"
    response = requests.get(url)
    data = response.json()

    return data


def get_macro_indicator_from_EOD(token, date_range):
    """compiles macroeconomic data from eod
    macroeconomic data: gdp, inflation, real interest rate
   
    Args:
        date_range (tuple): range of dates to extract data from
        token (str): eod api token
    
    Returns:
        pd.Dataframe: Dataframe containing the macroeconomic data within the date range
    """

    #get raw data from eod
    country_code = 'PHL'

    gdp_url = f"https://eodhistoricaldata.com/api/macro-indicator/{country_code}?api_token={token}&fmt=json&indicator=gdp_current_usd"
    infl_url = f"https://eodhistoricaldata.com/api/macro-indicator/{country_code}?api_token={token}&fmt=json&indicator=inflation_consumer_prices_annual"
    intrst_url = f"https://eodhistoricaldata.com/api/macro-indicator/{country_code}?api_token={token}&fmt=json&indicator=real_interest_rate"
    
    gdp_data = requests.get(gdp_url).json()
    infl_data = requests.get(infl_url).json()
    intrst_data = requests.get(intrst_url).json()

    gdp_data = pd.json_normalize(gdp_data)
    infl_data = pd.json_normalize(infl_data)
    intrst_data = pd.json_normalize(intrst_data)

    #identify start and end date
    start_date = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")

    #create empty lists to store data
    gdp_date = []
    gdp = []
    infl = []
    intrst = []

    #filter out data within the data range
    for ind in gdp_data.index:
        if start_date <= datetime.datetime.strptime(gdp_data['Date'][ind], "%Y-%m-%d") <= end_date:
            gdp_date.append(gdp_data['Date'][ind])
            gdp.append(gdp_data['Value'][ind])
            infl.append(infl_data['Value'][ind])
            intrst.append(intrst_data['Value'][ind])
    
    #convert into pd.Dataframe
    macro_data = pd.DataFrame({
        "date": gdp_date,
        "gdp": gdp,
        "inflation": infl,
        "real_interest_rate": intrst
    })
    
    return macro_data


def get_fundamental_data(stock_ticker, date_range):
    """Computes fundamental data for a specific stock. Also adds macro economic data. To be used for model training.
    fundamental data: gdp, inflation, real interest rate, returns on equity (roe), earnings per share (eps), price to earings ratio (p/e), psei returns
    
    Args:
        stock_ticker (str): stock ticker of the company to get data from
        date_range (tuple): range of dates to extract data from
    
    Returns:
        pd.Dataframe: Dataframe representing the fundamental indicators.
    """

    #get api key
    with open('keys/EOD_API_key.txt') as file:
        token = file.readline()
    
    #get trading dates and use it as the base pd.dataframe for fundamental data
    fundamental_data = pd.DataFrame(get_fundamental_trading_dates(stock_ticker, date_range, token))

    #macro data (yearly)
    fundamental_data['gdp'] = np.NAN
    fundamental_data['inflation'] = np.NAN
    fundamental_data['real_interest_rate'] = np.NAN

    #fill in macroeconomic data yearly
    macro_data = get_macro_indicator_from_EOD(token, date_range)
    for ind1 in macro_data.index:
        for ind2 in fundamental_data.index:
            if datetime.datetime.strptime(fundamental_data['date'][ind2], "%Y-%m-%d") <= datetime.datetime.strptime(macro_data['date'][ind1], "%Y-%m-%d"):
                fundamental_data.at[ind2, 'gdp'] = macro_data['gdp'][ind1]
                fundamental_data.at[ind2, 'inflation'] = macro_data['inflation'][ind1]
                fundamental_data.at[ind2, 'real_interest_rate'] = macro_data['real_interest_rate'][ind1]

    #company data (mixed)
    fundamental_data['roe'] = np.NAN
    fundamental_data['eps'] = np.NAN

    #get fundamental indicators from EOD
    raw_fundamental_data = get_fundamental_indicator_from_EOD(stock_ticker, token)

    #identify date range
    start_date = datetime.datetime.strptime(date_range[0], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(date_range[1], "%Y-%m-%d")
    
    #roe (yearly)
    roe = []

    #create empty lists to compute roe
    yearly_date = []
    share_holder_assets = []

    #filter necessary data to compute roe
    net_income_data = raw_fundamental_data['Financials']['Income_Statement']['yearly']
    share_holder_assets_data = raw_fundamental_data['Financials']['Balance_Sheet']['yearly']

    #compute roe
    for item, key in net_income_data.items():
        if start_date <= datetime.datetime.strptime(key['date'], "%Y-%m-%d") <= end_date:
            yearly_date.append(key['date'])
            roe.append(float(key['netIncome']))

    for item, key in share_holder_assets_data.items():
        if start_date <= datetime.datetime.strptime(key['date'], "%Y-%m-%d") <= end_date:
            share_holder_assets.append(float(key['totalStockholderEquity']))
    
    for ind1 in range(len(roe)):
        roe[ind1] = roe[ind1]/share_holder_assets[ind1]

    #fill in roe data yearly
    for ind1 in range(len(roe)):
        for ind2 in fundamental_data.index:
            if datetime.datetime.strptime(fundamental_data['date'][ind2], "%Y-%m-%d") <= datetime.datetime.strptime(yearly_date[ind1], "%Y-%m-%d"):
                fundamental_data.at[ind2, 'roe'] = roe[ind1]
    
    #eps (quarterly)

    #create empty lists to store eps data
    quarterly_date = []
    eps = []

    #filter eps data
    eps_data = raw_fundamental_data['Earnings']['History']
    for item, key in eps_data.items():
        if start_date <= datetime.datetime.strptime(key['date'], "%Y-%m-%d") <= end_date:
            quarterly_date.append(key['date'])
            eps.append(key['epsActual'])

    #fill in eps data quarterly
    for ind1 in range(len(eps)):
        for ind2 in fundamental_data.index:
            if datetime.datetime.strptime(fundamental_data['date'][ind2], "%Y-%m-%d") <= datetime.datetime.strptime(quarterly_date[ind1], "%Y-%m-%d"):
                fundamental_data.at[ind2, 'eps'] = eps[ind1]
    
    #propagate last available data
    fundamental_data.fillna(method='ffill', inplace=True)
    
    #p/e (daily)
    fundamental_data['p/e'] = np.NAN

    try:
        close = 'adjusted_close'
    except KeyError:
        close = 'close'

    #compute p/e
    for ind in fundamental_data.index:
        fundamental_data.at[ind, 'p/e'] = fundamental_data[close][ind]/fundamental_data['eps'][ind1]
    
    #remove unnecessary data
    fundamental_data.drop(columns=close, inplace=True)

    #psei returns (daily)
    psei_returns = get_psei_returns(date_range, token)

    #merge data sets
    fundamental_data = fundamental_data.join(psei_returns.set_index('date'), on='date',)

    return fundamental_data
#get_fundamental_data END

#data_processing START
def scale_data(data):
    """Scales the data so that there won't be errors with the power transformer
    Args:
        data (pd.DataFrame): The entire dataset.
    Returns:
        pd.DataFrame, pd.DataFrame: Scaled datasets.
    """	

    for indicator in data.columns:
         scale_factor = round(math.log10(data[f'{indicator}'].abs().max()))
         data[f'{indicator}'] = data[f'{indicator}'].apply(lambda x: float(Decimal(x) / Decimal(10 ** (scale_factor))))

    return data


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


def inverse_transform_data(data, scaler, col_names, feature="Stock Returns"):
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


def data_processing(technical_data, fundamental_data):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 8:2.
    Args:
        technical_data (pd.DataFrame): The dataset containing technical indicators.
        fundamental_data (pd.DataFrame): The dataset containing fundamental indicators.
    Returns:
        scaler, PowerTransformer: Scaler used in the power transform
        train, pd.DataFrame: The processed train dataset.
        test, pd.DataFrame: The processed test dataset.
        col_names, list: List of column names in the train and test datasets
    """	
    data = technical_data.join(fundamental_data.set_index('date'), on='date')
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    #remove date column
    data.drop(columns='date', inplace=True)

    #scale data
    scaled_data = scale_data(data)

    #split data into train and test
    train, test = train_test_split(scaled_data)
    
    #apply Yeo-Johnson Power Transfrom
    scaler, train, test, col_names = transform_data(train, test)

    return scaler, train, test, col_names


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
    stock_returns_index = train.columns.get_loc("log_return")

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
#data_processing END

def main():
    stock_ticker = 'BPI'
    scaler, train, test, col_names = data_processing(get_technical_data(stock_ticker, get_dates_five_years(testing=True)), get_fundamental_data(stock_ticker, get_dates_five_years(testing=True)))
    print(train, test)
    print(type(scaler))
    train_x, train_y, test_x, test_y = make_data_window(train, test)

if __name__ == '__main__':
    main()