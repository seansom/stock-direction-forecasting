import pandas as pd
import numpy as np
import tensorflow as tf
import datetime, requests, json, re

from decimal import Decimal
from eventregistry import *
from itertools import chain


def requests_get(url):
    """Wrapper function for the requests.get method. Implements a connection timeout
    error handling by restarting the GET request after 10 seconds.

    Args:
        url (str): The URL of the GET request.

    Returns:
        Response: The response of the GET request.
    """

    while True:
        try:
            return requests.get(url, timeout=10)
        except requests.exceptions.Timeout:
            print('Timeout. Restarting request...')
            continue


#get_technical_data START
def get_dates_seven_years(testing=False):
    """Returns a 2-item tuple of dates in yyyy-mm-dd format 7 years in between today.

    Args:
        testing (bool, optional): If set to true, always returns ('2015-04-13', '2022-04-13'). Defaults to False.

    Returns:
        tuple: (from_date, to_date)
    """

    if testing:
        return ('2015-04-13', '2022-04-13')

    # generate datetime objects
    date_today = datetime.datetime.now()
    date_seven_years_ago = date_today - datetime.timedelta(days=round(365.25 * 7))

    return (date_seven_years_ago.strftime('%Y-%m-%d'), date_today.strftime('%Y-%m-%d'))


def get_trading_dates(stock_ticker, date_range, token):
    """Returns a dataframe of all trading dates of a given stock in a given date range.

    Args:
        stock_ticker (str): The stock whose trading dates is needed.
        date_range (tuple): (from_date, to_date)
        token (str): The EOD API token used for authentication.

    Returns:
        pd.Dataframe: A dataframe of all the trading dates of a stock within the specified date range.
    """   

    exchange = 'PSE'
    url = f"https://eodhistoricaldata.com/api/eod/{stock_ticker}.{exchange}?api_token={token}&order=a&fmt=json&from={date_range[0]}&to={date_range[1]}"

    response = requests_get(url)
    data = response.json()

    # convert to pd.dataframe
    trading_dates = (pd.json_normalize(data))['date']

    return trading_dates


def get_technical_indicators(data):
    """Computes for technical indicators unavailable to EOD (e.g., A/D, ROC, WR).

    Args:
        data (pd.Dataframe): Dataframe containing dates and OHLCV stock data from EOD.

    Returns:
        pd.Dataframe: Dataframe containing dates, closing prices, and computed technical indicators.
    """    

    # get closing prices
    try:
        close = data['adjusted_close']
    except KeyError:
        close = data['close']

    data_len = len(close)

    closing = close

    # compute momentum values
    momentum_period = 5
    momentums = [np.NaN] * (momentum_period)
    
    for i in range(momentum_period, data_len):
        momentum = close[i] - close[i - 5]
        momentums.append(momentum)

    # compute rate of change values
    roc_period = 5
    rocs = [np.NaN] * (roc_period)

    for i in range(roc_period, data_len):
        roc = ((Decimal(close[i]) - Decimal(close[i - 5])) / Decimal(close[i - 5])) * 100
        rocs.append(roc)

    # compute William's %R indicator values
    wr_period = 5
    wr = [np.NaN] * (wr_period - 1)

    for i in range(wr_period, data_len + 1):
        wr_high = (data['high'][i - wr_period : i]).max()
        wr_low = (data['low'][i - wr_period : i]).min()
        wr_close = close[i - 1]

        if wr_low == wr_high:
            raise Exception(f"Error getting William's %R indicator. A period has the same highest and lowest price (zero division error).")
        
        curr_wr = Decimal(wr_high - wr_close) / Decimal(wr_high - wr_low)
        wr.append(curr_wr)

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

    # convert to dataframe
    technical_indicators = pd.DataFrame({
        'date' : data['date'],
        'closing' : closing,
        'momentum' : momentums,
        'roc' : rocs,
        'wr' : wr,
        'ad' : ad,
    })

    return technical_indicators


def get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, date_range):
    """Gets daily technical indicator data from EOD API.

    Args:
        indicator (str): The indicator for use in EOD API calls. (e.g., SMA)
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
        stock_ticker (str): The stock ticker being examined (e.g., BPI).
        date_range (tuple): A tuple of strings indicating the start and end dates for the requested data.

    Raises:
        Exception: Raises exception whenever np.NaN is present in the processed technical dataset.
        This can occur if the EOD API data is missing data for a technical indicator on a specific day.
        
    Returns:
        pd.Dataframe: Dataframe representing the dates, closing prices, and technical indicators.
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

    # compute returns and technical indicators not available on EOD
    technical_indicators = get_technical_indicators(data)
    data = data.merge(technical_indicators, on='date')

    # get sma values to be used for disparity 5 computations
    sma = get_technical_indicator_from_EOD('sma', 5, token, stock_ticker, exchange, (first_trading_day, last_trading_day))
    first_trading_day_index = data.index[data['date'] == first_trading_day].to_list()[0]

    # compute for disparity 5 values
    disparities = []
    for i in range(len(sma)):
        disparity = ((Decimal(data['close'][first_trading_day_index + i] - sma['sma'][i])) / Decimal(sma['sma'][i])) * 100
        disparities.append(disparity)

    temp_data = pd.DataFrame({
        'date' : sma['date'],
        'disparity' : disparities
    })

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
    EOD_indicators = [('stochastic', 5)]

    for indicator, period in EOD_indicators:
        indicator_data = get_technical_indicator_from_EOD(indicator, period, token, stock_ticker, exchange, (first_trading_day, last_trading_day))
        data = data.merge(indicator_data, on='date')

    data = data.merge(temp_data, on='date')

    # remove unneeded features/columns in dataframe
    data = data.drop(columns=['open', 'high', 'low', 'adjusted_close', 'volume', 'close'])

    if data.isnull().values.any():
        raise Exception(f'Null value found in technical dataset for {stock_ticker}')

    return data
#get_technical_data END


#news processing START
def load_json_to_dict(path_file_name):
    """Loads a dictionary stored in a json file.

    Args:
        path_file_name (str): The json file name or the path to it.

    Returns:
        dict: The dictionary stored in the json file.
    """

    file = open(path_file_name, "r")
    data = json.loads(file.read())
    file.close()

    return data


def get_news_from_API(stock_ticker, date_range, historical=False):
    """Gets news data from News API.

    Args:
        stock_ticker (str): The stock ticker being examined (e.g., BPI).
        date_range (tuple): A tuple of strings indicating the start and end dates for the requested data.
        historical (bool, optional): If set to true, access historical news data. Defaults to False.
    
    Returns:
        tuple: (list, dict)
            list: A list containing raw (complete with extras) news data.
            dict: A dictionary with dates as keys and the corresponding news data as values.
    """

    # related terms for each stock ticker (will be updated in the future to include all stocks in PSE)
    ticker_terms = {
        "ALI": ["Ayala Land"],
        "AP": ["Aboitiz Power"],
        "BPI": ["BPI", "Bank of the Philippine Islands"],
        "JFC": ["Jollibee Foods", "Jollibee"],
        "MER": ["Meralco", "Manila Electric Company"],
        "PGOLD": ["Puregold"],
        "SM": ["SM Investments"],
        "TEL": ["PLDT"]
        }

    if historical:
        with open('keys/News_API_paid_key.txt') as file:
            token = file.readline()

        er = EventRegistry(apiKey=token, allowUseOfArchive=True)
    else:
        with open('keys/News_API_free_key.txt') as file:
            token = file.readline()

        er = EventRegistry(apiKey=token, allowUseOfArchive=False)
    
    # adjust starting date by 1 day before
    start_date = datetime.datetime.strptime(date_range[0], '%Y-%m-%d')
    start_date = (start_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    q = QueryArticlesIter(
        dateStart=start_date,
        dateEnd=date_range[1],
        keywords=QueryItems.OR(ticker_terms[stock_ticker]), 
        sourceLocationUri=er.getLocationUri("Philippines"),
        keywordsLoc="body,title",
        ignoreKeywords=QueryItems.OR(["PBA", "basketball"]),
        ignoreConceptUri=QueryItems.OR([er.getConceptUri("PBA"), er.getConceptUri("basketball"), er.getConceptUri("Sports")]),
        ignoreCategoryUri=er.getCategoryUri("Sports"),
        isDuplicateFilter="skipDuplicates",
        hasDuplicateFilter="skipHasDuplicates"
        )

    raw_news = []
    news_dict = dict()
    for article in q.execQuery(er, sortBy="date", sortByAsc=True, maxItems=q.count(er)):
        raw_news.append(article)

        date = ""
        title = article["title"]
        body = article["body"]
        curr_news = [title, body]
        
        if ("dateTimePub" in article) and (article["dateTimePub"] != None):
            dateTime = article["dateTimePub"]
            dateTime = re.sub("[a-zA-Z]", "", dateTime)
            dateTime = datetime.datetime.strptime(dateTime, '%Y-%m-%d%H:%M:%S')
            # adjust dateTime by 8 hours from UTC+0 to UTC+8
            date = (dateTime + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
        else:
            dateTime = article["dateTime"]
            dateTime = re.sub("[a-zA-Z]", "", dateTime)
            dateTime = datetime.datetime.strptime(dateTime, '%Y-%m-%d%H:%M:%S')
            # adjust dateTime by 8 hours from UTC+0 to UTC+8
            date = (dateTime + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
        
        if (date < date_range[0]) or (date > date_range[1]):
            continue

        if date in news_dict:
            dict_val = news_dict[date]
            dict_val.append(curr_news)
            news_dict.update({date: dict_val})
        else:
            news_dict[date] = [curr_news]

    return raw_news, news_dict


def get_news(stock_ticker, date_range):
    """Builds a news dictionary for a given stock and date range.

    Args:
        stock_ticker (str): The stock ticker being examined (e.g., BPI).
        date_range (tuple): A tuple of strings indicating the start and end dates for the requested data.
        
    Returns:
        dict: A dictionary with dates as keys and the corresponding news headlines as values.
    """

    news_path = "sentiment data/" + str(stock_ticker) + "/news.json"

    if os.path.exists(news_path):
        news_data = load_json_to_dict(news_path)
        dates_list = list(news_data.keys())

        with open('keys/EOD_API_key.txt') as file:
            token = file.readline()

        trading_days = get_trading_dates(stock_ticker, date_range, token)
        trading_days = trading_days.tolist()

        # if currently saved news data is sufficient (i.e., last required day is earlier than last saved day)
        if trading_days[-1] <= dates_list[-1]:
            news_dict = dict()
            for day in trading_days:
                curr_day_news = []
                
                if day in news_data:
                    for news in news_data[day]:
                        curr_day_news.append(news[0])

                news_dict[day] = curr_day_news

        # if currently saved news data is insufficient
        else:
            more_news_data = get_news_from_API(stock_ticker, (dates_list[-1], date_range[1]), historical=False)
            latest_news = more_news_data[1]

            news_dict = dict()
            for day in trading_days:
                curr_day_news = []
                
                if day in news_data:
                    for news in news_data[day]:
                        curr_day_news.append(news[0])

                elif day in latest_news:
                    for news in latest_news[day]:
                        curr_day_news.append(news[0])

                news_dict[day] = curr_day_news
            
    return news_dict


def clean_text(text):
    """Cleans an input text (sentence).

    Args:
        text (str): The text to be cleaned.

    Returns:
        string: The cleaned text.
    """

    # remove all non-letters (punctuations, numbers, etc.)
    processed_text = re.sub("[^a-zA-Z]", " ", text)
    # remove single characters
    processed_text = re.sub(r"\s+[a-zA-Z]\s+", " ", processed_text)
    # remove multiple whitespaces
    processed_text = re.sub(r"\s+", " ", processed_text)

    # clean more
    processed_text = re.sub(r"\s+[a-zA-Z]\s+", " ", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text)
    
    return processed_text


def text_to_int(tokenizer, texts, max_seq_length, is_train=True):
    """Converts a set of texts (sentences) into their numerical representations.

    Args:
        tokenizer (keras.Tokenizer): The created tokenizer object.
        texts (list): The list of texts (sentences) to be converted.
        max_seq_length (int): The length of the longest sentence in the dataset.
        is_train (bool): If set to True, currently converting the training inputs dataset. Defaults to True.

    Returns:
        list: A list of lists wherein each element represents the numerical representation of a specific text (sentence).
    """
    
    if is_train:
        # convert texts to numerical sequences
        sequences = tokenizer.texts_to_sequences(texts)
        # pad sequences
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding="post")
    
    else:
        # convert texts to numerical sequences
        sequences = []
        for text in texts:
            text_list = text.split()
            sequence = []
            for i in range(len(text_list)):
                #tokenizer.word_index is a dictionary with words as keys and integers as values
                if text_list[i] not in tokenizer.word_index:
                    sequence.append(0)
                else:
                    sequence.append(tokenizer.word_index[text_list[i]])
                
                # truncate sentences longer than max_seq_length
                if len(sequence) == max_seq_length:
                    break
            
            sequences.append(sequence)

        # pad sequences
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding="post")

    return sequences


def news_processing(train_x_news, test_x_news, max_news_count):
    """Function to process news data to convert each headline into its numerical representation.

    Args:
        train_x_news, list: A list of lists which contain news headlines per day for the training date range.
        test_x_news, list: A list of lists which contain news headlines per day for the test date range.
        max_news_count, int: The most number of headlines on a given day over the whole date range. 

    Returns:
        vocab_size, int: The number of words in the vocabulary.
        tokenizer: The constructed tokenizer during conversion of texts into their numerical representation. 
        train_x_news, list: A list of lists which contain news headlines (numerical representation) per day for the training date range.
        test_x_news, list: A list of lists which contain news headlines (numerical representation) per day for the test date range.
        max_seq_length, int: The length of the numerical sequence equivalent of the longest headline.
    """

    flattened_train_x_news = list(chain.from_iterable(train_x_news))
    flattened_test_x_news = list(chain.from_iterable(test_x_news))

    cleaned_headlines_train = []
    for headline in flattened_train_x_news:
        cleaned_headlines_train.append(clean_text(headline).lower())

    cleaned_headlines_test = []
    for headline in flattened_test_x_news:
        cleaned_headlines_test.append(clean_text(headline).lower())

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(cleaned_headlines_train)
    vocab_size = len(tokenizer.word_index) + 1
    
    max_seq_length = 0
    cleaned_headlines = cleaned_headlines_train + cleaned_headlines_test
    for headline in cleaned_headlines:
        if len(headline.split()) > max_seq_length:
            max_seq_length = len(headline.split())

    pad = [0] * max_seq_length
    
    train_sequences = []
    for window in train_x_news:    
        sequences = list(text_to_int(tokenizer, window, max_seq_length, is_train=True))
        while len(sequences) < max_news_count:
            sequences.append(np.array(pad))
        train_sequences.append(sequences)

    test_sequences = []
    for window in test_x_news:
        sequences = list(text_to_int(tokenizer, window, max_seq_length, is_train=False))
        while len(sequences) < max_news_count:
            sequences.append(np.array(pad))
        test_sequences.append(sequences)

    train_x_news, test_x_news = np.array(train_sequences), np.array(test_sequences)

    return vocab_size, tokenizer, train_x_news, test_x_news, max_seq_length
#news processing END


# data processing START
def make_targets(data, time_steps=1):
    """Function that makes targets.

    Args:
        data (pd.DataFrame): The input data.
        time_steps (int, optional): How many time steps should each target correspond to. Defaults to 1.

    Returns:
        list: A list containing the targets.
    """

    targets = []

    for i in range(time_steps, len(data)):
        if data['closing'][i] >= data['closing'][i - 1]:
            targets.append([1, 0])
        else:
            targets.append([0, 1])

    return targets


def make_windows(data, time_steps=1):
    """Function that makes windows.

    Args:
        data (pd.DataFrame): The input data.
        time_steps (int, optional): How many time steps should be in each data window. Defaults to 1.

    Returns:
        list: A list containing the windows.
    """

    data = data.to_numpy()
    data_len = data.shape[0]

    windows = []
    for i in range(data_len):
        if (i + time_steps) < data_len:
            windows.append([data[j, 1:] for j in range(i, i + time_steps)])

    return windows


def data_processing(technical_data, news_data, drop_col=None, time_steps=1):
    """Splits a dataset into training and testing samples.
    The train and test data are split with a ratio of 9:1.

    Args:
        technical_data (pd.DataFrame): The dataset containing technical indicators.
        news_data (dict): A dictionary with dates as keys and the corresponding news headlines as values.
        drop_col (list, optional): The list of dropped features or columns in the dataset. Defaults to None.
        time_steps (int, optional): How many time steps should be in each data window. Defaults to 1.

    Returns:
        train_x, list: Array representing the training inputs dataset (technical indicators only).
        train_y, list: Array representing the training targets dataset.
        test_x, list: Array representing the testing inputs dataset (technical indicators only).
        test_y, list: Array representing the testing targets dataset.
        train_x_news, list: A list of lists which contain news headlines per day for the training date range.
        test_x_news, list: A list of lists which contain news headlines per day for the test date range.
        max_news_count, int: The most number of headlines on a given day over the whole date range.
    """

    data = technical_data
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    data = data.merge(news_data, on='date')

    #remove date column
    data.drop(columns=['date'], inplace=True)

    if drop_col is not None:
        data.drop(columns=drop_col, inplace=True)

    # raise exception if missing/null value is found in the combined dataset
    if data.isnull().values.any():
        raise Exception(f'Null value found in combined dataset.')

    news_windows = make_windows(data, time_steps)

    indices_to_remove = []
    max_news_count = 0
    for i in range(len(news_windows)):
        news_count = len(news_windows[i][-1][-1]) 
        if news_count == 0:
            indices_to_remove.append(i)
        if news_count > max_news_count:
            max_news_count = news_count

    #remove date column
    data.drop(columns=['news'], inplace=True)

    windows = make_windows(data, time_steps)
    targets = make_targets(data, time_steps)

    cleaned_windows = []
    cleaned_targets = []
    cleaned_news_windows = []
    for i, _ in enumerate(windows):
        if i not in indices_to_remove:
            cleaned_windows.append(windows[i])
            cleaned_targets.append(targets[i])
            cleaned_news_windows.append(news_windows[i][-1][-1])

    test_len = len(cleaned_windows) * 1 // 10
    train_x, test_x = cleaned_windows[:-test_len], cleaned_windows[-test_len:]
    train_y, test_y = cleaned_targets[:-test_len], cleaned_targets[-test_len:]
    train_x_news, test_x_news = cleaned_news_windows[:-test_len], cleaned_news_windows[-test_len:]

    return train_x, train_y, test_x, test_y, train_x_news, test_x_news, max_news_count


def get_dataset(stock_ticker, date_range=None, time_steps=1, drop_col=None):
    """Function that gets, processes, and returns a specified stock's dataset 
    at a particular date_range from the APIs.

    Args:
        stock_ticker (str): The stock whose dataset is needed.
        date_range (tuple, optional): (from_date, to_date). Defaults to None.
        time_steps (int, optional): The number of time_steps or window size of the dataset. Defaults to 1.
        drop_col (list, optional): The list of dropped features or columns in the dataset. Defaults to None.
    
    Returns:
        train_x, np.array: Array representing the training inputs dataset (technical indicators only).
        train_y, np.array: Array representing the training targets dataset. 
        test_x, np.array: Array representing the testing inputs dataset (technical indicators only).
        test_y, np.array: Array representing the testing targets dataset.
        vocab_size, int: The number of words in the vocabulary.
        tokenizer: The constructed tokenizer during conversion of texts into their numerical representation. 
        train_x_news, list: A list of lists which contain news headlines (numerical representation) per day for the training date range.
        test_x_news, list: A list of lists which contain news headlines (numerical representation) per day for the test date range.
        max_seq_length, int: The length of the numerical sequence equivalent of the longest headline.
    """

    if date_range == None:
        date_range = get_dates_seven_years(testing=True)

    technical_data = get_technical_data(stock_ticker=stock_ticker, date_range=date_range)
    
    news_data = get_news(stock_ticker=stock_ticker, date_range=date_range)

    date = list(news_data.keys())
    news = list(news_data.values())

    news_data = pd.DataFrame({
        'date' : date,
        'news' : news
    })

    train_x, train_y, test_x, test_y, train_x_news, test_x_news, max_news_count = data_processing(technical_data, news_data, drop_col, time_steps)
    
    vocab_size, tokenizer, train_x_news, test_x_news, max_seq_length = news_processing(train_x_news, test_x_news, max_news_count)

    train_x = np.asarray(train_x).astype(np.float32)
    train_y = np.asarray(train_y).astype(np.float32)
    test_x = np.asarray(test_x).astype(np.float32)
    test_y = np.asarray(test_y).astype(np.float32)

    return train_x, train_y, test_x, test_y, vocab_size, tokenizer, train_x_news, test_x_news, max_seq_length
#data processing END


def main():
    stock_ticker = 'TEL'
    
    train_x, train_y, test_x, test_y, vocab_size, tokenizer, train_x_news, test_x_news, max_seq_length = get_dataset(stock_ticker, date_range=None, time_steps=5, drop_col=None)
    
    print(train_x.shape)
    print(test_y.shape)
 

if __name__ == '__main__':
    main()