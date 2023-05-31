import numpy as np
import pandas as pd
import nltk
# nltk.download('all')
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Sequential, load_model, model_from_config
import math
from nn import *

# fungsi untuk memanggil model LSTM
def get_model():
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()
    return model

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

def makeVect(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model.wv[i])        
    vec = np.divide(vec,noOfWords)
    return vec


def getVects(essays, model, num_features):
    c=0
    essay_vecs = np.zeros((len(essays),num_features),dtype="float32")
    for i in essays:
        essay_vecs[c] = makeVect(i, model, num_features)
        c+=1
    return essay_vecs

def MakeModel(pathFile, user_id):
  # ambil data dari xlxs
  df = pd.read_excel(pathFile, engine='openpyxl')
  df.drop(columns='nomor pertanyaan', inplace=True,axis=1)
  X = df['jawaban'].tolist()

  train_sents=[]
  for i in X:
    train_sents+=essay2word(i)
  
  # new df
  res = []
  sc = []
  for z in range(len(train_sents)):
      a = ""
      d = []
      for x in train_sents[z]:
          a +=x+" "
          res.append(a.strip())
          d.append(x)
          c = math.ceil(0.1*round(len(d)*(100/len(train_sents[z]))))
          sc.append(c)
  
  df = pd.DataFrame({"Key":res, "Score":sc})

  # buat dataset
  y = df['Score']
  df.drop('Score',inplace=True,axis=1)
  X = df

  train_e = X['Key'].tolist()
  test_e = X['Key'].tolist()

  # ubah kalimat menjadi kata-kata
  train_sents=[]
  test_sents=[]
  for i in train_e:
    train_sents+=essay2word(i)

  for i in test_e:
      test_sents+=essay2word(i)

# buat model word2vec
  path_save = "/MyFolder/About Program/Tutorial/python/flask"
  #Training Word2Vec model
  num_features = 300 
  min_word_count = 2
  num_workers = 4
  context = 5
  downsampling = 1e-3

  model = Word2Vec(train_sents, 
                  workers=num_workers, 
                  vector_size=num_features, 
                  min_count = min_word_count, 
                  window = context, 
                  sample = downsampling)

  # model.init_sims(replace=True)
  # model.wv.wmdistance()
  model.wv.save_word2vec_format(path_save+'/'+user_id+'/word2vecmodel.bin', binary=True)

  clean_train=[]
  for i in train_e:
      clean_train.append(sent2word(i))
  training_vectors = getVects(clean_train, model, num_features)

  clean_test=[] 

  for i in test_e:
      clean_test.append(sent2word(i))
  testing_vectors = getVects(clean_test, model, num_features)

  training_vectors = np.array(training_vectors)
  testing_vectors = np.array(testing_vectors)

  # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
  training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
  testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))
  lstm_model = get_model()

  lstm_model.fit(training_vectors, y, batch_size=64, epochs=300)

  lstm_model.save(path_save+'/'+user_id+'/final_lstm.h5')

  return 200


