# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:10:11 2018
@author: Titli
"""
import pandas as pd
import html, re
import nltk
import pickle
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

path = "E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\"

def save_obj(objname, fname):
    pickle.dump( objname, open(fname, "wb" ) )
    
def load_obj(fname):
        return pickle.load( open(fname, "rb" ) )
    
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

def glovetoword2vec_conversion(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)
    
def loadGloveModel(modelloc):
    print ("Loading Glove Model")
    model = KeyedVectors.load_word2vec_format(modelloc, binary=False)
    return model

def avg_equdim_corpus(corpus_dict):
    averaged_dict={}
    for k,v in corpus_dict.items():
        tmp = []
        for i in v:
            tmp.append(i)
        tmp=np.array(tmp)
        averaged_dict[k]=np.mean(tmp, axis=0)
        #if (averaged_dict[k].shape[0] != 100):
            #print (k, averaged_dict[k].shape[0])
    return averaged_dict

def dict_to_array(dict):
    arr = []
    i=0
    for k, v in dict.items():
        i = i+1
        arr.append(v)
    arr = np.array(arr)
    #save_obj(processed_corpus, path+'train_data.txt')
    return arr
'''
def GaussianNaiveBayes(train_data, train_label):
    gnb = GaussianNB(priors=None)
    y_pred = gnb.fit(train_data, train_label).predict(train_data)
    #miss = y_pred - train_label
    return y_pred

def BernoulliNaiveBayes(train_data, train_label):
    clf = BernoulliNB()
    clf.fit(train_data, train_label)
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    y_pred = clf.predict(train_data)
    return y_pred

def random_forest(train_data, train_label):
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) # Instantiate model with 1000 decision trees
    rf.fit(train_data, train_label)
    y_pred = rf.predict(train_data)
    return y_pred
'''
def dict_to_array2d(dict):
    s = pd.DataFrame.from_dict(dict)
    return s

def dict_to_array1d(dict):
    arr = []
    i=0
    for k, v in dict.items():
        i = i+1
        arr.append(v)
    arr = np.array(arr)
    return arr
## main function
def main():
    
    train_file = path+"2018-EI-reg-En-train\\EI-reg-En-full-train.csv"
    test_file = path+"2018-EI-reg-En-test\\EI-reg-En-part-test.csv"
    dev_file = path+"2018-EI-reg-En-dev\\EI-reg-En-full-dev.csv"
    
    ### Part 1: Preprocessing the corpus
    data_dict, label_dict, intensity_dict = preprocessing(test_file)
    save_obj(data_dict, path+'test_data_dict.txt')
    save_obj(label_dict, path+'test_label_dict.txt')
    save_obj(intensity_dict, path+'test_intensity_dict.txt')
    
    ### Part 2: Feature Selection
    # i) Glove Conversion
    glove_file_path = 'E:\\Studies\\Ph.D\\Sem 4\\CSCE 619\\SemEval2018Task\\glove.twitter.27B\\'
    glove_input_file = glove_file_path+'glove.twitter.27B.100d.txt'
    word2vec_output_file = glove_file_path+'word2vec.twitter.27B.100d.txt'
    
    glovetoword2vec_conversion(glove_input_file, word2vec_output_file) #1
    print ("Glove to Word2Vec conversion Done!")
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False) #2
    print ("Glove model loaded")
    #print ("Angry=",glove_model['angry'])
    
    # ii) representing each word as a glove word embedding 
    new_corpus = load_obj('test_data_dict.txt')
    gloved_data_dict = {} # dictionary of key=id, value=sentence as vector of words
    for k,v in new_corpus.items():
        valuevectList = []
        present_words = filter(lambda x: x in glove_model.vocab, v)
        for item in present_words:
            valuevectList.append(glove_model[item])
        gloved_data_dict[k]=valuevectList
    print ("Glove representation done!")
    save_obj(gloved_data_dict, path+'gloved_test_data_dict.txt') ## saving corpus in the form of dict, key=id, value=each tweet as glove-embedded words
    
    # iii) Averaging for equidim
    gloved_data = load_obj('gloved_test_data_dict.txt')
    equidim_data_dict = avg_equdim_corpus(gloved_data)  ## dictionary
    save_obj(equidim_data_dict, path+'equidim_test_data_dict.txt')
    print (equidim_data_dict)
    print("Avraging done!")
    
    # iv) format change: dict to matrix
    data = dict_to_array2d(load_obj('equidim_test_data_dict.txt'))
    label = dict_to_array1d(load_obj('test_label_dict.txt'))
    intensity = dict_to_array1d(load_obj('test_intensity_dict.txt'))
    
    data.to_csv(path+'test_data.csv')
    np.savetxt(path+'test_label.csv', label)
    np.savetxt(path+'test_intensity.csv', intensity)
    
    #train = np.loadtxt(path+'train_data.csv')
    #print (train.shape)
    ### Part 3: Classify
    # Split the data into training and testing sets
    ## for cross validation
    #train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size = 0.25, random_state = 42)
    '''
    # i)Naive Bayes
    Gaussian_y_pred = GaussianNaiveBayes(data, label)
    acc = (1 - ((label != Gaussian_y_pred).sum() / data.shape[0]))*100
    print("Number of mislabeled points out of a total %d points : %d"
          % (data.shape[0],(label != Gaussian_y_pred).sum()))
    miss = Gaussian_y_pred-label
    print("GaussianNB Accuracy is:", acc)

    # ii) BernoulliNB
    Bernoulli_y_pred = BernoulliNaiveBayes(data, label)
    acc = (1 - ((label != Bernoulli_y_pred).sum() / data.shape[0]))*100
    print("BernoulliNB Accuracy is:", acc)
    
    # iii) Random Forest
    RF_y_pred = random_forest(data, label)
    acc = (1 - ((label != RF_y_pred).sum() / data.shape[0]))*100
    print("random_forest Accuracy is:", acc)
    '''
if __name__=="__main__":
    main()
    