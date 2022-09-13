
# Import Libraries
import json
from xml.dom.xmlbuilder import DocumentLS

import nltk
import pickle
import tensorflow as tf

tf.autograph.set_verbosity(1)
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

lemmatizer = WordNetLemmatizer()
import random

import numpy as np
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('./datasets/intents.json').read()
intents = json.loads(data_file)
print('--------------------------------------------------------------------------------')
# Tokenizing is the most basic and first thing you can do on text data. Tokenizing is the process of breaking the whole text into small parts like words.

# TOkenization
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add document in the corpus(collection of text--> body)
        documents.append((w, intent['tag']))

        #add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


def done():
    print('so far so good')




done()

