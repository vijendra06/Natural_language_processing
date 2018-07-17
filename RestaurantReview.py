#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 23:15:22 2018

@author: vijendrasharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t' , quoting=3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,1000):
    review= re.sub('[^a-zA-z]', ' ', dataset.values[i,0] )
    review= review.lower()
    review= review.split()
    ps= PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)


    
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features= 1500)
x= cv.fit_transform(corpus).toarray()
y= dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain, ytest= train_test_split(x,y,test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
clf= GaussianNB()
clf.fit(xtrain, ytrain)

ypred= clf.predict(xtest)
acc= clf.score(xtest, ytest)
