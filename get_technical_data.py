import pandas as pd
import numpy as np
from decimal import Decimal
import datetime, requests, json, math, sys, os


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


def main():
    print(get_technical_data('BPI', get_dates_five_years(testing=True)))


if __name__ == '__main__':
    main()