import numpy as np
import pandas as pd
import tensorflow as tf
import re, json, os, shutil, zipfile

from gensim import models
from statistics import mean
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense


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


def preprocess(dataframe):
    """Preprocess data which includes cleaning news headlines, mapping string sentiment labels into their integer equivalent,
    and splitting into train and test datasets.

    Args:
        dataframe (pd.Dataframe): Dataframe representing the news headlines and their corresponding sentiment labels.

    Returns:
        np.array (4): Arrays representing the train and test datasets.
    """

    #clean all headlines then lowercase
    clean_headlines = []
    for headline in dataframe["Headline"]:
        clean_headline = clean_text(headline)
        clean_headlines.append(clean_headline.lower())
        
    sentiment_mapping = {
        "negative" : 0,
        "neutral" : 1,
        "positive" : 2
    }

    #convert string sentiment to integer equivalent
    sentiments = []
    for sentiment in dataframe["Sentiment"]:
        sentiments.append(sentiment_mapping[sentiment])
    
    #split dataset to train and test datasets
    train_x, test_x, train_y, test_y = train_test_split(clean_headlines, sentiments, train_size=0.8, shuffle=False)
    
    return np.array(train_x), np.array(test_x), np.array(train_y), np.array(test_y)


def make_tokenizer(train_x):
    """Creates a tokenizer object using the training inputs dataset.

    Args:
        train_x (np.array): Array representing the training inputs dataset.

    Returns:
        keras.Tokenizer: The created tokenizer object.
    """

    #create tokenizer object
    tokenizer = Tokenizer()
    #look at list of texts to count frequency of unique words, most common asigned integer value of 1
    tokenizer.fit_on_texts(train_x)
    
    return tokenizer


def text_to_int(tokenizer, vocab, texts, is_train):
    """Converts a set of texts (sentences) into their numerical representations.

    Args:
        tokenizer (keras.Tokenizer): The created tokenizer object.
        vocab (dict): A dictionary with words as keys and their integer equivalent as values.
        texts (list): The list of texts (sentences) to be converted.
        is_train (bool): If set to True, currently converting the training inputs dataset.

    Returns:
        list: A list of lists wherein each element represents the numerical representation of a specific text (sentence).
    """

    #transform sentences into integer representation
    if is_train:
        sequences = tokenizer.texts_to_sequences(texts)

    else:
        sequences = []
        for text in texts:
            text_list = text.split()
            sequence = []
            for word in text_list:
                if word not in vocab:
                    sequence.append(0)
                else:
                    sequence.append(vocab[word])

            sequences.append(sequence)
    
    return sequences


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


def make_cnn_model(hps, vocab_size, max_seq_length, word_embeddings):
    """Builds and compiles a CNN model.

    Args:
        hps (dict): A dictionary representing model hyperparameters.
        vocab_size (int): The number of words in the vocabulary.
        max_seq_length (int): The length of the longest sequence (sentence).
        word_embeddings (array): An array wherein each row represents a vector representation of a specific word in the vocabulary.

    Returns:
        keras.model: The CNN model built and compiled.
    """

    model = Sequential()

    filters = hps["filters"]
    kernel_size = hps["kernel_size"]
    rate = hps["rate"]
    
    model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[word_embeddings], input_length=max_seq_length, trainable=False))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(rate=rate))
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train_cnn_model(model, train_x, train_y):
    """Trains the CNN model based on the given training dateset.

    Args:
        model (keras.model): The CNN model built and compiled.
        train_x (np.array): Array representing the training inputs dataset.
        train_y (np.array): Array representing the training targets dataset.

    Returns:
        model (keras.model): The trained CNN model.
    """

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')
    model.fit(train_x, train_y, epochs=50, verbose=0, validation_split=0.25, callbacks=[early_stopping_callback])

    print("Training Done")
    
    return model


def get_accuracy(model, test_x, test_y):
    """Evaluates the trained CNN model on the test dataset.

    Args:
        model (keras.model): The trained CNN model.
        test_x (np.array): Array representing the test inputs dataset.
        test_y (np.array): Array representing the test targets dataset.

    Returns:
        float: The accuracy of the trained CNN model.
    """

    score = model.evaluate(test_x, test_y, verbose=0)
    
    return round(score[1], 6)


def experiment(train_x, test_x, train_y, test_y, hps, raw_embeddings):
    """Function that creates and evaluates a single CNN model.

    Args:
        train_x (np.array): Array representing the training inputs dataset.
        test_x (np.array): Array representing the test inputs dataset.
        train_y (np.array): Array representing the training targets dataset.
        test_y (np.array): Array representing the test targets dataset.
        hps (dict): A dictionary representing model hyperparameters.
        raw_embeddings (dict): A dictionary with words as keys and their vector representation as values.

    Returns:
        model, keras.model: The created CNN model.
        accuracy, float: The accuracy of the created CNN model.
        tokenizer, keras.Tokenizaer: The tokenizer object created based on the training inputs dataset.
        max_seq_length, int: The length of the longest sequence (sentence).
    """

    # create tokenizer object
    tokenizer = make_tokenizer(train_x)

    # create vocabulary
    vocab = tokenizer.word_index
    vocab_size = len(vocab) + 1
    
    # convert sentences into numerical sequences
    train_sequences = text_to_int(tokenizer, vocab, train_x, True)
    test_sequences = text_to_int(tokenizer, vocab, test_x, False)
    
    # pad sequences up to the length of the longest sequence
    max_seq_length = max(np.max(list(map(lambda x: len(x), train_sequences))), np.max(list(map(lambda x: len(x), test_sequences))))
    train_x = pad_sequences(train_sequences, maxlen=max_seq_length, padding="post")
    test_x = pad_sequences(test_sequences, maxlen=max_seq_length, padding="post")
    
    #get word embeddings (vector representing each word)
    word_embeddings = get_weight_matrix(raw_embeddings, vocab)

    # build, compile, and train the CNN model
    model = train_cnn_model(make_cnn_model(hps, vocab_size, max_seq_length, word_embeddings), train_x, train_y)
    
    return model, get_accuracy(model, test_x, test_y), tokenizer, max_seq_length


def main():
    # load embeddings
    raw_embeddings = load_embedding("word2vec.txt")

    # load data
    data = pd.read_csv("all-data.csv", names=["Sentiment", "Headline"], encoding="latin-1")
    
    # hps
    hps = {"filters": 256, "kernel_size": 3, "rate": 0.3}

    # number of models built
    best_acc = 0
    repeats = 50
    accuracies = []
    for i in range(repeats):
        print("Model " + str(i + 1) + "/" + str(repeats))

        # shuffle and split data
        curr_data = shuffle(data)
        train_x, test_x, train_y, test_y = preprocess(curr_data)

        curr_model, curr_acc, curr_tokenizer, max_seq_length = experiment(train_x, test_x, train_y, test_y, hps, raw_embeddings)
        accuracies.append(curr_acc)

        if curr_acc > best_acc:
            best_model = curr_model
            best_acc = curr_acc
            best_tokenizer = curr_tokenizer
            best_data = curr_data

    print()
    print("Accuracies:", accuracies)
    print("Mean accuracy: " + str(mean(accuracies)))
    print("Best accuracy: " + str(best_acc))
    print("================================================================================")

    # load saved model
    if os.path.exists("best sentiment model"):
        shutil.rmtree("best sentiment model")

    with zipfile.ZipFile("best sentiment model.zip", "r") as zip_ref:
        zip_ref.extractall()
        
    loaded_model = tf.keras.models.load_model("best sentiment model")

    shutil.rmtree("best sentiment model")

    # load saved data for saved model 
    loaded_data = pd.read_csv("all-data-best.csv", names=["Sentiment", "Headline"], encoding="latin-1")
    train_x, test_x, train_y, test_y, = preprocess(loaded_data)

    # load saved vocab for saved model
    file = open("sentiment data/vocab.json", "r")
    loaded_vocab = json.loads(file.read())
    file.close()

    # get accuracy of saved model
    test_sequences = text_to_int(None, loaded_vocab, test_x, False)
    test_x = pad_sequences(test_sequences, maxlen=max_seq_length, padding="post")
    loaded_accuracy = get_accuracy(loaded_model, test_x, test_y)
    
    print()
    print("Best accuracy among built models:", best_acc)
    print("Accuracy of previously saved model:", loaded_accuracy)
    print()

    message = "Overwrite previously saved files? ('Y' if yes, 'N' if no): "
    reply = input(message)
    while (reply != "Y" and reply != "N"):
        reply = input(message)
    
    if reply == "Y":
        print()
        print("Now overwriting files ...")

        # overwrite saved model
        best_model.save("temp/best sentiment model") 
        shutil.make_archive("best sentiment model", "zip", "temp")
        shutil.rmtree("temp")

        # overwrite saved data
        best_data.to_csv("all-data-best.csv", header=False, index=False)

        # overwrite saved vocab
        file = open("sentiment data/vocab.json", "w")
        json.dump(best_tokenizer.word_index, file)
        file.close()

        print("Success")


if __name__ == '__main__':
    main()