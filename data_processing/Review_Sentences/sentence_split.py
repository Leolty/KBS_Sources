# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:50:22 2022

@author: Leo
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
import nltk.data

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
     cleanr = re.compile('<..*?>-•·*')
     text = re.sub(cleanr, '', text)
     
     return text

if __name__ == '__main__':
    # Load data
   # df = pd.read_excel('TETC.xlsx','1')
   df = pd.read_excel('E:\\dataset\\Google_Drive\\Google_Drive_Reviews_Origin.xlsx','Sheet1')
   df = df.dropna()
   df = df.reset_index(drop=True)
   # df['title_content'] = df['title'] +". "+ df['content']
   texts = df['content']
   lists = []
   
   # split review into sentence
   for text in texts:
       s_list = nltk.sent_tokenize(text)
       lists.append(s_list)
       
   for s_list in lists:
       index = -1
       for sentence in s_list:
           index = index + 1
           s_list[index] = clean_text(sentence)
           
           
   for s_list in lists:
       index = -1
       for sentence in s_list:
           index = index + 1
           if sentence.isspace() == 1 or sentence == " ":
               del s_list[index]
               print("del")
               index = index - 1
               
   dic = {"date":[],"rating":[],"sentence":[]}    
   i = -1
   for s_list in lists:
       i = i+1
       j = -1
       for sentence in s_list:
           j = j + 1
           dic["rating"].append(df['rating'][i])
           dic["date"].append(df['date'][i])
           dic["sentence"].append(sentence)
           
   dataFrame = pd.DataFrame.from_dict(dic)
   df = df.dropna()
   df = df.reset_index(drop=True)
   dataFrame.to_excel("Google_Drive_Review_Sentences.xlsx")