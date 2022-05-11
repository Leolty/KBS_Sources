# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:20:43 2021

@author: Tianyang Liu
"""

import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.models import doc2vec, ldamodel
from gensim import corpora
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from gensim.models import Word2Vec
import gensim
import xlwings as xw
from scipy.linalg import norm
import glob,os
import logging
from gensim.models.word2vec import LineSentence

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
 

def remove_noise(sentence):
    result = ''
    poster = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words('english'))
    wordlist = re.sub(r"\n|(\\(.*?){)|}|[!$%^&*#()_+|~\-={}\[\]:\";'<>?,.\/\\]|[0-9]|[@]", ' ', sentence) # remove punctuation
    wordlist = re.sub('\s+', ' ', wordlist) # remove extra space
    wordlist_normal = [poster.stem(word.lower()) for word in wordlist.split()] # restore word to its original form (stemming)
    wordlist_normal = [lemmatizer.lemmatize(word, pos='v') for word in wordlist_normal] # restore word to its root form (lemmatization)
    wordlist_clean = [word for word in wordlist_normal if word not in stopword_set] # remove stopwords
    result = ' '.join(wordlist_clean)
    
    return result

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


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
    # batch read files
    path=r'E:\dataset\data_processing\Review_Sentences'
    # xlsx
    file=glob.glob(os.path.join(path, "*.xlsx"))
    print(file)
    dl= []
    for f in file:
        # read every excel file
        dl.append(pd.read_excel(f))  
    # combine
    df=pd.concat(dl)     
    
    df = df[['sentence', 'label']].dropna()
    
    # filter informative sentences
    data = df['sentence'].loc[df['label'] == 1]
    
    # clen sentence
    data = data.apply(lambda s: clean_text(s))
    doclist = data.values.tolist()
    
    doclist = [s.lower() for s in doclist]
    
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
    
#    # 保存至新的数组
#    new_texts = []
#    for review in result:
#        temp = []
#        if len(review) != 0:
#            for word in review:
#                temp.append(word[0])
#        new_texts.append(temp)
    
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
            
    
    listFile = open("E:\dataset\WordVectorModel\informative_reviews.txt","w")
    

    for review in doclist:
        listFile.write(review+'\n')
    
    listFile.close()
    
    # 设置输出日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 直接用gemsim提供的API LineStence 去读取txt文件
    sentences = LineSentence("E:\dataset\WordVectorModel\informative_reviews.txt")

    # 训练模型，词向量的长度设置为300， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
    model = gensim.models.Word2Vec(sentences, vector_size=300, epochs=8,sg=1)  
    model.wv.save_word2vec_format("E:\dataset\WordVectorModel/word2vec_model" + ".bin", binary=True) 

    # 加载bin格式的模型
    wordVec = gensim.models.KeyedVectors.load_word2vec_format("E:\dataset\WordVectorModel\word2vec_model_lemmated.bin", binary=True)
    wordVec2 = gensim.models.KeyedVectors.load_word2vec_format("E:\dataset\WordVectorModel\word2vec_model.bin", binary=True)
    print(wordVec.most_similar(positive=['i'], topn=10))
    print(wordVec2.most_similar(positive=['i'], topn=10))