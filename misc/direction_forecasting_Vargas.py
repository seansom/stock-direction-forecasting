# This version of the direction forecasting model recreates the model from the study by Vargas, dos Anjos, Bichara, and Evsukoff
# Title: Deep Learning for Stock Market Prediction Using Technical Indicators and Financial News Articles

# Model Type: CNN-LSTM (CNN to handle news articles while LSTM to handle technical indicators)
# Model Inputs: 
    # Technical: ['k_values', 'd_values', 'momentum', 'roc', 'wr', 'ad', 'disparity']
    # News articles (headlines)
# Model Outputs: closing as either [1, 0] or [0, 1] (upward or downard)
# Loss: Not specified (used Binary Crossentropy instead)
# Optimizer: SGD (0.1 learning rate, 0.9 momentum)
# Epochs: Not specified (used 100 instead)

# Note: Minor changes were made to the embedding layer.

import numpy as np

from tensorflow import keras
from gensim import models
from statistics import mean, stdev
from data_processing_Vargas import get_dataset


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


def load_embedding(filename):
    """Loads embedding from a given file.

    Args:
        filename (str): The name of file.

    Returns:
        dict: A dictionary with words as keys and their vector representation as values.
    """

    word2vec = models.KeyedVectors.load_word2vec_format(filename, binary = True)
    word2vec_dict = dict(zip(word2vec.key_to_index.keys(), word2vec.vectors))
        
    return word2vec_dict
 

def get_weight_matrix(embedding, vocab):
    """Creates weight matrix from a given embedding.

    Args:
        embedding (dict): A dictionary with words as keys and their vector representation as values.
        vocab (dict): A dictionary with words as keys and their integer equivalent as values.

    Returns:
        array: An array wherein each row represents a vector representation of a specific word in the vocabulary.
    """

	#total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
	#define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 300))
	#store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector

    return weight_matrix


def make_cnn_lstm_model(train_x, train_y, vocab_size, word_embeddings, num_news, train_x_news_dict, epochs=100):
    """Builds and returns a CNN-LSTM model based on a given training dataset.

    Args:
        train_x (np.array): Array representing the training inputs dataset (technical indicators only).
        train_y (np.array): Array representing the training targets dataset.
        vocab_size (int): The number of words in the vocabulary.
        word_embeddings (array): An array wherein each row represents a vector representation of a specific word in the vocabulary.
        num_news (int): The number of news per day.
        train_x_news_dict (dict): A dictionary of inputs such that the values are the list of all the ith day news titles.
        epochs (int, optional): The number of epochs that the model will be trained for. Defaults to 100.

    Returns:
        keras.model: The CNN-LSTM model built and trained.
    """

    # technical indicators
    # prepare technical input
    technical_input = keras.layers.Input(shape=(train_x.shape[1:]), name='technical_input')
    
    # pass technical input to lstm layer
    technical_lstm = keras.layers.LSTM(units=128)(technical_input)

    # news data
    # prepare news inputs
    inputs = []
    for i in range(num_news):
        title = 'title_' + str(i)
        inputs.append(keras.layers.Input(shape=(None,), name=title))
    
    # pass inputs to cnn
    feats = []
    for i in range(len(inputs)):
        make_embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=300, weights=[word_embeddings], trainable=False)(inputs[i])
        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3)(make_embedding)
        conv2 = keras.layers.Conv1D(filters=64, kernel_size=4)(conv1)
        conv3 = keras.layers.Conv1D(filters=64, kernel_size=5)(conv2)
        pool = keras.layers.MaxPooling1D(pool_size=2)(conv3)
        activation = keras.layers.Activation(keras.activations.relu)(pool)
        drop = keras.layers.Dropout(rate=0.5)(activation)
        feats.append(drop)
    
    # concatenate before passing to lstm layer directly connected to cnn
    pre_lstm = keras.layers.concatenate(feats)

    # pass to lstm layer
    news_lstm = keras.layers.LSTM(units=128)(pre_lstm)

    # contanate technical and news data models
    pre_softmax = keras.layers.concatenate([technical_lstm, news_lstm])

    # pass to dense layer for prediction
    final_pred = keras.layers.Dense(units=2, activation='softmax')(pre_softmax)

    # specify model inputs and outputs
    model_inputs = [technical_input] + inputs
    cnn_lstm_model = keras.models.Model(inputs=model_inputs, outputs=final_pred)

    print_train_progress_callback = CustomCallback(epochs)
    optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    cnn_lstm_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.BinaryAccuracy()])
    
    # specify data name inputs
    data_inputs = {'technical_input' : train_x}
    data_inputs.update(train_x_news_dict)
    cnn_lstm_model.fit(data_inputs, train_y, epochs=epochs, verbose=0, callbacks=[print_train_progress_callback])
    
    return cnn_lstm_model


def forecast_cnn_lstm_model(model, test_x, test_x_news_dict, max_seq_length):
    """Forecasts future values using a model and test input dataset.
    
    Args:
        model (keras.model): The built and trained CNN-LSTM model used for forecasting.
        test_x (np.array): Array representing the testing inputs dataset (technical indicators only).
        test_x_news_dict (dict): A dictionary of inputs such that the values are the list of all the ith day news titles.
        max_seq_length (int): The length of the numerical sequence equivalent of the longest headline.
    
    Returns:
        list: A list of the forecasted future values.
    """	
    predictions = []

    test_len = test_x.shape[0]
    test_timesteps = test_x.shape[1]
    test_features = test_x.shape[2]

    for i in range(test_len):
        curr_progress = round(((i + 1) / test_len) * 100, 2)
        print(f'Prediction Progress: {curr_progress} %', end='\r')

        model_input = {}
        
        technical_input = (test_x[i, :, :]).reshape(1, test_timesteps, test_features)
        model_input['technical_input'] = technical_input

        for key in test_x_news_dict.keys():
            model_input[key] = test_x_news_dict[key][i].reshape(1, max_seq_length)

        prediction = model.predict(model_input)
        
        predictions.append(prediction)
    print()

    return predictions


def get_cnn_lstm_model_perf(predictions, actuals):
    """Calculates performance metrics of a model given its predictions
    and actual future values.

    Args:
        predictions (np.array): A list of forecasted future values.
        actuals (np.array): A numpy array of actual future values.

    Returns:
        dict: A dictionary containing difference performance metrics of a model.
    """	

    predictions_len = len(predictions)
    actuals = actuals.tolist()

    total_ups = 0
    total_downs = 0
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(predictions_len):
        predicted = predictions[i].tolist()
        prediction = predicted[0].index(max(predicted[0]))
        actual = actuals[i].index(max(actuals[i]))

        # calculate number of total actual upward and downward directions
        if actual == 0:
            total_ups += 1
        else:
            total_downs += 1
            
        # calculate true positives, true negatives, false positives, and false negatives
        if prediction == 0 and actual == 0:
            tp += 1
        elif prediction == 1 and actual == 1:
            tn += 1
        elif prediction == 0 and actual == 1:
            fp += 1
        else:
            fn += 1

    # calculate directional accuracy, upward directional accuracy, and downward directional accuracy
    da = (tp + tn) / (tp + tn + fp + fn)
    uda = (tp / (tp + fp)) if (tp + fp) else 1
    dda = (tn / (tn + fn)) if (tn + fn) else 1

    # store performance metrics in a dictionary
    return {"total_ups":total_ups, "total_downs":total_downs, "tp":tp, "tn":tn, "fp":fp, "fn":fn, "da":da, "uda":uda, "dda":dda}


def experiment(stock_ticker, time_steps, raw_embeddings, date_range=None, drop_col=None):
    """Function that creates and evaluates a single model. Returns the performance metrics of the created model.
    
    Args:
        stock_ticker (string): The target stock to be predicted.
        time_steps (int): The number of timesteps in the data window inputs.
        raw_embeddings (dict): A dictionary with words as keys and their vector representation as values.
        date_range (tuple, optional): (from_date, to_date). Defaults to None.
        drop_col (list, optional): The features or columns to be dropped in the model. Defaults to None.
    
    Returns:
        dict: A dictionary of the performance metrics of the created model.
    """

    train_x, train_y, test_x, test_y, vocab_size, tokenizer, train_x_news, test_x_news, max_seq_length = get_dataset(stock_ticker, date_range=date_range, time_steps=time_steps, drop_col=drop_col)
    word_embeddings = get_weight_matrix(raw_embeddings, tokenizer.word_index)

    # number of news per day
    num_news = train_x_news.shape[1]
    
    # make a dictionary of inputs such that the values are the list of all the ith news titles
    train_x_news_dict = {}
    for i in range(num_news):
        col_name = 'title_' + str(i)
        col_news = []
        for j in range(train_x_news.shape[0]):
            col_news.append(train_x_news[j][i])
        
        train_x_news_dict[col_name] = np.array(col_news)

    # create, compile, and fit a cnn-lstm model
    cnn_lstm_model = make_cnn_lstm_model(train_x, train_y, vocab_size, word_embeddings, num_news, train_x_news_dict, epochs=100)

    test_x_news_dict = {}
    for i in range(num_news):
        col_name = 'title_' + str(i)
        col_news = []
        for j in range(test_x_news.shape[0]):
            col_news.append(test_x_news[j][i])
        
        test_x_news_dict[col_name] = np.array(col_news)

    # get the model predictions
    predictions = forecast_cnn_lstm_model(cnn_lstm_model, test_x, test_x_news_dict, max_seq_length)

    # get model performance statistics
    perf = get_cnn_lstm_model_perf(predictions, test_y)

    return perf


def main():
    raw_embeddings = load_embedding("word2vec.txt")

    # how many models built (min = 2)
    repeats = 10
    time_steps = 5

    tickers = ["ALI", "AP", "BPI", "JFC", "MER", "PGOLD", "SM", "TEL"]
    for stock_ticker in tickers:
        print("===================================================")
        performances = []

        for i in range(repeats):
            print(f"Experiment {i + 1} / {repeats}")
            perf = experiment(stock_ticker, time_steps, raw_embeddings, date_range=None, drop_col=None)
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

        # print average accuracies of the built models
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