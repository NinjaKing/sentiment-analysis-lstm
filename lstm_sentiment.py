
# coding: utf-8

# In[24]:


import pandas as pd
import pickle
import numpy as np
import os
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Bidirectional, Convolution2D, MaxPooling2D
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

import html, re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')


# In[3]:


path = "data/"
GLOVE_DIR = "data/glove.twitter.27B/"


# In[4]:


input_dim = 100


# In[5]:


def ReadCSV(datafile, labelfile):
    inputdata = pd.io.parsers.read_csv(open(datafile, "r"),delimiter=",")
    data = inputdata.as_matrix()
    #data = (tmpdata/255.0)-0.5
    label = np.loadtxt(open(labelfile, "rb"),delimiter=",")
    return data, label


# In[6]:


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def stopwordsremoval(sentence):
    stopwords_removed = [word for word in sentence.split(' ') if word not in stopwords.words('english')]
    return stopwords_removed

def clean_str(string):
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    #string = string.replace("_NEG", "")
    #string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string) #removes @---, 
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ,", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)
    return stopwordsremoval(strip_punctuation(string.strip().lower()))

def preprocessing(train_file): ## we will return everything as dictionaries
    corpus_dict = {}
    intensity_dict = {}
    affect_dict = {}
    df=pd.read_csv(train_file,encoding='utf-8')
    id = df['ID'] # not used
    train_sentences=df['Tweet']
    intensity_scores=df['Intensity Score']
    affect_dimension = df['Affect Dimension']
    
    for (k1,v1),(k2,v2),(k3,v3) in zip(train_sentences.iteritems(), intensity_scores.iteritems(), affect_dimension.iteritems()):
        intensity_dict[k2] = v2
        affect_dict[k3] = v3
        # adding processed tweets in a dict
        sentence = sent_tokenize(v1) # sentence tokenize, list of sentences
        processed_tweet = []
        for sen in sentence:
            sen1=""
            sen1 = clean_str(sen)
            processed_tweet = processed_tweet+sen1
        corpus_dict[k1]=processed_tweet 
    return corpus_dict,affect_dict,intensity_dict


# In[7]:


def one_hot_encoding(y):
    y = to_categorical(y)
    return y[:,1:] #remove extra zero column at the first


# In[8]:


#def dict_to_array(dic):
#    return [v for _, v in dic.items()]


# In[9]:


def prepare_data(data_file_name):
    data_path = path + data_file_name
    processed_data_path = path + 'processed-' + data_file_name
    # check if file is processed
    if os.path.isfile(processed_data_path):
        print("Processed file:", data_file_name)
        df = pd.read_csv(processed_data_path)
        inputs = [str(x).split() for x in df.iloc[:, 1].values]
        labels = df.iloc[:, 0].values
        return (inputs, labels)
    
    # preprocessing and save into csv file
    print("Preprocessing data file:", data_file_name)
    inputs, labels, _ = preprocessing(data_path)

    # convert dict into array
    inputs = dict_to_array(inputs)
    labels = dict_to_array(labels)

    # save into csv
    df_save = pd.DataFrame({'x': [' '.join(x) for x in inputs], 'label': labels})
    df_save.to_csv(processed_data_path, encoding='utf-8', index=False)
    
    return (inputs, labels)    


# In[10]:


# convert glove to w2v
glove_input_file = GLOVE_DIR + 'glove.twitter.27B.100d.txt'
word2vec_output_file = GLOVE_DIR + 'word2vec.twitter.27B.100d.txt'
"""
glove2word2vec(glove_input_file, word2vec_output_file)
print("Glove to Word2Vec conversion Done!")
"""

word2vec = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print("Load word2vec done!")


# In[11]:


# read data file
train_data, train_label = prepare_data('EI-reg-En-full-train.csv')
dev_data, dev_label = prepare_data('EI-reg-En-full-dev.csv')
test_data, test_label = prepare_data('EI-reg-En-part-test.csv')

print("Train:", len(train_data), len(train_label))
print("Val:", len(dev_data), len(dev_label))
print("Test:", len(test_data), len(test_label))


# In[12]:


input_data = np.concatenate((train_data, dev_data, test_data))
max_sequence_length = max([len(x) for x in input_data])
print("Max sequence length:", max_sequence_length)


# In[13]:


# embedding data
def embedding(data, max_len):
    data_eb = np.zeros((len(data), max_len, input_dim))
    for i in range(len(data)):
        vec = []
        for j, token in enumerate(data[i]):
            if token in word2vec:
                data_eb[i][-len(data[i]) + j] = word2vec[token]            
    return data_eb

train_data = embedding(train_data, max_sequence_length)
dev_data = embedding(dev_data, max_sequence_length)
test_data = embedding(test_data, max_sequence_length)

print("Train embedding:", train_data.shape, train_label.shape)
print("Dev embedding:", dev_data.shape, dev_label.shape)
print("Test embedding:", test_data.shape, test_label.shape)


# In[14]:


# convert label to one-hot vector
labels = np.concatenate((train_label, dev_label, test_label))
number_classes = len(np.unique(labels))
print("Number class:", number_classes)
y_oh = one_hot_encoding(labels)

train_label = y_oh[:train_label.shape[0]]
dev_label = y_oh[train_label.shape[0]:train_label.shape[0] + dev_label.shape[0]]
test_label = y_oh[-test_label.shape[0]:]

print("One-hot encoded:", train_label.shape, dev_label.shape, test_label.shape)


# In[27]:


def compile_model(input_dim, latent_dim, num_class):
    '''Create model

    Args:
        input_dim (int): dim of embedding vector (glove dimension)
        latent_dim (int): dim of output from LSTM layer
        num_class (int): number output class
    '''
    inputs = Input(shape=(None, input_dim))
    lstm = Bidirectional(LSTM(latent_dim))(inputs)
    drop = Dropout(0.5)(lstm)
    #flat = Flatten()(drop)
    out = Dense(num_class, activation='softmax')(drop)

    model = Model(inputs, out)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


# In[28]:


# create lstm model
model = compile_model(input_dim, 128, number_classes)


# In[29]:


epochs = 100
batch_size = 128

checkpointer = ModelCheckpoint(filepath='twitter-emotion.h5', verbose=1, save_best_only=True)
model.fit(train_data, train_label, validation_data=(dev_data, dev_label), callbacks=[checkpointer], 
          shuffle=True, epochs=epochs, batch_size=batch_size, verbose=2)


# In[24]:


# test
model.evaluate(test_data, test_label)

