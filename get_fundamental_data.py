import requests, json
import pandas as pd
import numpy as np
import datetime
import math
from decimal import Decimal

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

    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={date_range[0]}&to={date_range[1]}"
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

def main():
    stock_ticker = 'BPI'
    fundamental_data = get_fundamental_data(stock_ticker, get_dates_five_years(testing=True))
    
    #convert to csv to check
    fundamental_data.to_csv('fundamental_data.csv')

if __name__ == '__main__':
    main()