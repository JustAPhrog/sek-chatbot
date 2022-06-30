import logging
import random
import string
from pathlib import Path

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense, Dropout

# Required packages
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

class Chatbot:
    logger = logging.getLogger('chatbot')

    def __init__(self):
        self.intents = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.doc_x = []
        self.doc_y = []

    def create_model(self, input_shape, output_shape):
        # the deep learning model
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=input_shape, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(output_shape, activation="softmax"))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=["accuracy"])
        print(self.model.summary())

    def load_model(self, model_path):
        if Path(model_path).exists() and Path(model_path).is_dir():
            self.model = keras.models.load_model(model_path)
            return True
        self.logger.error('Cannot load model!')
        return False

    def prepare_model(self, epochs, trainig_data=None):
        if trainig_data is None:
            trainig_data = self.generate_training_data()
        # split the features and target labels
        train_x = np.array(list(trainig_data[:, 0]))
        train_y = np.array(list(trainig_data[:, 1]))
        # defining some parameters
        input_shape = (len(train_x[0]),)
        output_shape = len(train_y[0])
        self.create_model(input_shape, output_shape)
        self.model.fit(x=train_x, y=train_y, epochs=epochs, verbose=1)

    def prepare_data(self, data):
        # Loop through all the intents
        # tokenize each pattern and append tokens to words, the patterns and
        # the associated tag to their associated list
        intents = 0
        for intent in data["intents"]:
            intents += 1
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.doc_x.append(pattern)
                self.doc_y.append(intent["tag"])

            # add the tag to the classes if it's not there already
            if intent["tag"] not in self.classes:
                self.classes.append(intent["tag"])
        # print(f'{intents};{patterns}')
        self.intents = intents
        # lemmatize all the words in the vocab and convert them to lowercase
        # if the words don't appear in punctuation
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in string.punctuation]
        # sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

    def generate_training_data(self):
        # list for training data
        training = []
        out_empty = [0] * len(self.classes)
        # creating the bag of words model
        for idx, doc in enumerate(self.doc_x):
            bow = []
            text = self.lemmatizer.lemmatize(doc.lower())
            for word in self.words:
                bow.append(1) if word in text else bow.append(0)
            # mark the index of class that the current pattern is associated
            # to
            output_row = list(out_empty)
            output_row[self.classes.index(self.doc_y[idx])] = 1
            # add the one hot encoded BoW and associated classes to training
            training.append([bow, output_row])
        # shuffle the data and convert it to an array
        random.shuffle(training)
        return np.array(training, dtype=object)

    def predict_class(self, text):
        bow = self.bag_of_words(text, self.words)
        result = self.model.predict(np.array([bow]), verbose=0)[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

        y_pred.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in y_pred:
            return_list.append(self.classes[r[0]])
        # print(return_list)
        return return_list

    def clean_text(self, text):
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)


if __name__ == "__main__":
    pass
