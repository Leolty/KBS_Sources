# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:23:56 2022

@author: Tianyang Liu
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from gensim.models import Word2Vec
import gensim
from scipy.linalg import norm
import xlwings as xw
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    # clean datas
     # clean the new line
     text = text.replace('\n', " ")  
     # clean the url
     # text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)
     text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
     # clean the email address
     text = re.sub(r"[\w]+@[\.\w]+", "", text)
     # clean the digits
     text = re.sub(r"[0-9]", "", text)
     # clean the special charactors
     text = re.sub('[^A-Za-z0-9]+', " ", text)
     # clean the words length less than 2
     # text = ' '.join(word for word in text.split() if len(word) > 2)
     return text
 

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

if __name__ == '__main__':
    
    # read data
    df = pd.read_excel('E:\sci_source_code\data_processing\Review_Sentences\Instagram_Review_Sentences.xlsx','Sheet1')
    df = df[['date','rating','sentence','label']].dropna()
    df = df[['date','rating','sentence']].loc[df['label'] == 1]
   
    # write informative data to excel
    df.to_excel('E:\sci_source_code\similarity_calculation\Instagram_Informative_Reviews.xlsx')
   
    data = df['sentence']
    # clen sentence
    data = data.apply(lambda s: clean_text(s))
    doclist = data.values.tolist() 
   
    # filter stopwords
    en_stops = set(stopwords.words('english')) 
    texts = [[word for word in doc.lower().split() if word not in en_stops] for doc in doclist]
   
    # filter n,v, and adj

    # JJ, JJR, JJS: 形容词，形容词比较级，形容词最高级
    # NN, NNS, NNP, NNPS： 名词，名词复数，专有名词，专有名词复数
    # VB, VBD, VBG, VBP, VBN, VBZ: 动词，动词过去式，动词现在分词，动词过去分词，动词现在式非第三人称时态，动词现在式第三人称时态
    corps = set(['JJ','JJR', 'JJS',
             'NN', 'NNS', 'NNP', 'NNPS',
             'VB','VBD', 'VBG','VBP', 'VBN', 'VBZ'])
   
    result = []
    for text in texts:
        result.append(pos_tag(text))
        
    for review in result:
        # 从后向前删除不符合条件的 防止溢出
        for i in range(len(review)-1,-1,-1):
            if review[i][1] not in corps:
                del review[i]
    
    # 词性还原
    wnl = WordNetLemmatizer()

    texts_lemmated = []    
    for review in result:
        temp = []
        if len(review) != 0:
            for word, tag in review:
                if tag.startswith('NN'): # noun
                    temp.append(wnl.lemmatize(word, pos='n'))
                elif tag.startswith('VB'): # verb
                    temp.append(wnl.lemmatize(word, pos='v'))
                elif tag.startswith('JJ'): # adj
                    temp.append(wnl.lemmatize(word, pos='a'))
        texts_lemmated.append(temp)
                    
    sentences = [' '.join(text) for text in texts_lemmated]
    
    # load word2vec model
    wordvec = gensim.models.KeyedVectors.load_word2vec_format("E:\sci_source_code\word2vec_model\word2vec_model.bin", binary=True)
    # wordvec = gensim.models.KeyedVectors.load_word2vec_format("E:\sci_source_code\word2vec_model\word2vec_model_lemmated.bin", binary=True)
    index2word_set = set(wordvec.index_to_key)
   
    changelog =  "new Messenger feature"
    changelog = [word for word in changelog.lower().split() if word not in en_stops]
    changelog = ' '.join(changelog)
    print(changelog)
   
    changelog_vec = avg_feature_vector(changelog, model=wordvec, num_features=300, index2word_set=index2word_set)
    sim = []
    for sentence in sentences:
        sentence_vec = avg_feature_vector(sentence, model=wordvec, num_features=300, index2word_set=index2word_set)
        sim.append(np.dot(sentence_vec, changelog_vec) / (norm(sentence_vec) * norm(changelog_vec)))
       # sim.append(cosine_similarity(sentence_vec, changelog_vec))
       
    file = open("E:\sci_source_code\similarity_calculation\Instagram_Sim_Result\Instagram_Messenger.txt","w")
    

    for s in sim:
        file.write(str(s)+'\n')
    
    file.close()