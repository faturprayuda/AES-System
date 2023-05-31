import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
# from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
from os import path
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import json
from bson import json_util


def sent2word(x):
    stop_words = set(stopwords.words('indonesian'))
    x = re.sub("[^A-Za-z]", " ", x)
    x.lower()
    filtered_sentence = []
    words = x.split()
    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def essay2word(essay):
    essay = essay.strip()
    # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = nltk.sent_tokenize(essay)
    final_words = []
    for i in raw:
        if(len(i) > 0):
            final_words.append(sent2word(i))
    return final_words


def makeVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model[i])        
    vec = np.divide(vec,noOfWords)
    return vec


def getVecs(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVec(i, model, num_features)
        c+=1
    return essay_vecs


def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4,
              input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model


def convertToVec(text, user_id):
    content = text
    if len(content) > 20:
        num_features = 300
        model = KeyedVectors.load_word2vec_format(
            user_id+"/word2vecmodel.bin", binary=True)
        clean_test_essays = []
        clean_test_essays.append(sent2word(content))
        testDataVecs = getVecs(clean_test_essays, model, num_features)
        testDataVecs = np.array(testDataVecs)
        testDataVecs = np.reshape(
            testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

        lstm_model = load_model(
            user_id+"/final_lstm.h5")
        preds = lstm_model.predict(testDataVecs)
        if (np.isnan(preds)):
            return str(0)
        else:
            # print('real score : ' + str(preds[0][0]))
            return str(round(preds[0][0]))
    else:
        return str(0)


def getFile(path_file, user_id):
    df = pd.read_excel(path_file, engine='openpyxl')
    df.dropna(axis=0, how='all', inplace=True)
    df.drop(columns='Timestamp', inplace=True, axis=1)
    temp_data = []

    for i in df:
        temp_data.append(df[i])

    data = []
    for i in range(len(temp_data[0])):
        tmp_separate = []
        for j in temp_data:
            tmp_separate.append(j[i])
        data.append(tmp_separate)

    key_arr = []
    for i in range(len(data)):
    # for i in range(1):
        key = {}
        key['nama'] = data[i][0]
        key['kelas'] = data[i][1]
        key['absen'] = str(data[i][2])
        key['jawaban1'] = convertToVec(data[i][3], user_id)
        key['jawaban2'] = convertToVec(data[i][4], user_id)
        key['jawaban3'] = convertToVec(data[i][5], user_id)
        key['jawaban4'] = convertToVec(data[i][6], user_id)
        key['jawaban5'] = convertToVec(data[i][7], user_id)
        key_arr.append(key)

    return key_arr
