# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:00:08 2022

@author: Leo
"""

from preprocessing import *
from sklearn import naive_bayes
from helper import get_instance_words,get_all_instance_words
from Semi_EM_NB import Semi_EM_MultinomialNB
from sklearn import metrics
from performance_metrics import get_accuracy
from performance_metrics import get_f_measure
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
from sklearn.externals import joblib


def main():
    training_data_list = ["combine_info.txt","combine_non-info.txt"]
    training_data_info = "combine_info.txt"
    training_data_noninfo = "combine_non-info.txt"
    
    trainU_data = "sk_unlabeled.txt"
    
    # get training data
    mapping = extract_words_and_add_to_dict(training_data_list)
    reverse_mapping = get_reverse_mapping(mapping)
    training_data0 = get_data([training_data_info], mapping)
    trainY = np.ones(training_data0.shape[0], dtype=int)
    
    training_data1 = get_data([training_data_noninfo], mapping)
    trainY = np.append(trainY,np.zeros(training_data1.shape[0],dtype=int))
    
    trainX = np.append(training_data0, training_data1, axis=0)
    
    trainU = get_data([trainU_data], mapping)
    print(trainU.shape)
    
    clf = Semi_EM_MultinomialNB()
    clf.fit(trainX, trainY, trainU)
    
    # joblib.dump(clf, 'filter_info_model.pkl')
    
    df = pd.read_excel('E:\dataset\data_processing\Review_Sentences\Google_Drive_Review_Sentences.xlsx', 'Sheet1')
    
    sentences = df['sentence']
    ratings = df['rating']
    
    sentences = sentences.values.tolist()
    ratings = ratings.values.tolist()
    for rating in range(0,len(ratings)):
        if ratings[rating] == 1:
            ratings[rating] = 'ratingone'
        elif ratings[rating] == 2:
            ratings[rating] = 'ratingtwo'
        elif ratings[rating] == 3:
            ratings[rating] = 'ratingthree'
        elif ratings[rating] == 4:
            ratings[rating] = 'ratingfour'
        elif ratings[rating] == 5:
            ratings[rating] = 'ratingfive'
    
    f = open('filter_data\Google_Drive.txt', 'w')
    
    for i in range(0,len(sentences)):
        f.write('len'+' '+ratings[i]+' '+ sentences[i]+'\n')
    
    f.close()
    
    pred_data = "filter_data\Google_Drive.txt"
    pred_data0 = get_data([pred_data], mapping)
    
    # get_all_instance_words(reverse_mapping,pred_data0,"helper.txt") 
    result = clf.predict(pred_data0)
    
    f = open('filter_data\Google_Drive_Result.txt','w')
    
    for j in range(0,len(result)):
        f.write(str(result[j])+'\n')
    
    f.close()
     
if __name__ == '__main__':
	main()

