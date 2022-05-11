# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:55:56 2022

@author: Leo
"""


from sklearn.utils import shuffle
import pandas as pd
from sklearn.externals import joblib
from preprocessing import *

df = pd.read_excel('E:\dataset\data_processing\Review_Sentences\Spotify_Review_Sentences.xlsx', 'Sheet1')
    
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
print(ratings)
    
f = open('filter_data\Spotify.txt', 'w')
    
for i in range(0,len(sentences)):
    f.write('len'+' '+ratings[i]+' '+ sentences[i]+'\n')
    
f.close()
    
pred_data = "filter_data\Spotify.txt"
pred_data0 = get_data([pred_data], mapping)
    
clf = joblib.load('filter_info_model.pkl')
result = clf.predict(pred_data0)
    
f = open('filter_data\Spotify_Result.txt','w')
    
for j in range(0,len(result)):
    f.write(str(result[j])+'\n')
    
f.close()