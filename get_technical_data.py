import pandas as pd
import numpy as np
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


def get_technical_indicators(data):
    """Computes for log stock returns and technical indicators unavailable to EOD (e.g., A/D, CMF, WR).
    Args:
        data (pd.Dataframe): Dataframe containing dates and OHLCV stock data from EOD.

    Returns:
        pd.Dataframe: Dataframe containing dates, log stock returns and technical indicators.
    """    

    # get closing prices from data
    try:
        close = data['adjusted_close']
    except KeyError:
        close = data['close']

    # compute log stock returns
    stock_returns = [np.NaN]
    for i in range(1, len(close)):
        stock_return = math.log(close[i] / close[i - 1])
        stock_returns.append(stock_return)

    # convert to dataframe
    returns = pd.DataFrame({
        'date': data['date'],
        'log_return': stock_returns
    })

    return returns


def get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, date_range):
    """Gets daily technical indicator data from EOD API.

    Args:
        indicator (str): The indicator for use in EOD API calls. (e.g., rsi)
        period (str): The period used in computing the technical indicator.
        token (str): The EOD API key or token.
        stock_ticker (str): The stock ticker being examined (e.g., BPI).
        exchange (str): The stock exchange where the stock is being traded (e.g., PSE)
        date_range (tuple): A tuple of strings indicating the start and end dates for the requested data.

    Returns:
        pd.Dataframe: Dataframe containing dates and specified technical indicator data
    """

    # compute an adjusted from or start date for the API
    original_from_date = datetime.datetime.strptime(date_range[0],'%Y-%m-%d')
    adjusted_from_date = original_from_date - datetime.timedelta(days=50)
    
    url = f"https://eodhistoricaldata.com/api/technical/{stock_ticker}.{exchange}?order=a&fmt=json&from={adjusted_from_date.strftime('%Y-%m-%d')}&to={date_range[1]}&function={indicator}&period={period}&api_token={token}"

    response = requests.get(url)
    data = response.json()

    # convert to pd.dataframe
    data = pd.json_normalize(data)

    # remove rows with dates earlier than wanted from date
    for index, row in data.iterrows():

        # date of current row entry
        curr_date = datetime.datetime.strptime(row['date'],'%Y-%m-%d')

        # remove unneeded earlier row entries
        if curr_date < original_from_date:
            data.drop(index, inplace=True)

        else:
            break

    # reset indices after dropping rows
    data = data.reset_index(drop=True)

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

    # compute an adjusted from or start date for the API. Extra days worth of data is included
    # to compute for log stock returns and technical indicators with extended periods
    original_from_date = datetime.datetime.strptime(date_range[0],'%Y-%m-%d')
    adjusted_from_date = original_from_date - datetime.timedelta(days=21)

    # get API key/token from txt file
    with open('keys/EOD_API_key.txt') as file:
        token = file.readline()

    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={adjusted_from_date.strftime('%Y-%m-%d')}&to={date_range[1]}"

    response = requests.get(url)
    data = response.json()

     # convert to pd.dataframe
    data = pd.json_normalize(data)

    # compute returns and technical indicators not available on EOD
    technical_indicators = get_technical_indicators(data)
    data = data.merge(technical_indicators, on='date')

    # remove rows with dates earlier than wanted from date
    for index, row in data.iterrows():
        curr_date = datetime.datetime.strptime(row['date'],'%Y-%m-%d')

        if curr_date < original_from_date:
            data.drop(index, inplace=True)

        else:
            break

    # reset indices after dropping rows
    data = data.reset_index(drop=True)

    # get available technical indicators from API
    atr = get_technical_indicator_from_EOD('atr', 14, token, stock_ticker, exchange, date_range)
    data = data.merge(atr, on='date')

    rsi = get_technical_indicator_from_EOD('rsi', 14, token, stock_ticker, exchange, date_range)
    data = data.merge(rsi, on='date')

    cci = get_technical_indicator_from_EOD('cci', 20, token, stock_ticker, exchange, date_range)
    data = data.merge(cci, on='date')

    adx = get_technical_indicator_from_EOD('adx', 14, token, stock_ticker, exchange, date_range)
    data = data.merge(adx, on='date')

    # remove unneeded features/columns in dataframe
    data = data.drop(columns=['open', 'high', 'low', 'adjusted_close', 'volume', 'close'])

    if data.isnull().values.any():
        raise Exception(f'Null value found in technical dataset for {stock_ticker}')

    return data


def main():
    print(get_technical_data('TEL', get_dates_five_years()))


if __name__ == '__main__':
    main()