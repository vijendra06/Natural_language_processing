#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:34:09 2018

@author: vijendrasharma
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier


from nltk.classify import ClassifierI
from statistics import mode

class voteclassifier(ClassifierI):
    def __init__ (self, *classifiers):
        self._classifiers= classifiers
        
    def classify(self, features):
        votes= []
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
            
        return mode(votes)
    
    def confidence(self, features):
        votes= []
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
            
            
        choice= votes.count(mode(votes))
        conf= choice/ len(votes)
        return conf
        
        
#creates 2d array.. one column for list of words and other for category
documents= [ (list( movie_reviews.words(fileid)), category) 
             for category in movie_reviews.categories() 
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
print(documents[100])
allwords= []
#converts all words in movie_reviews.words(): to lower case 
for w in movie_reviews.words():
    allwords.append(w.lower())
 # sorts all words on basis of frequency   
allwords= nltk.FreqDist(allwords)
'''print( allwords.most_common(25))
    
print( allwords["stupid"])'''
#extracts 3500 most common words
wordfeatures= list(allwords.keys())[:3500]



# if particular word is sentence= true else false
def findfeatures(document):
    words= set(document)
    features= {}
    for w in wordfeatures:
        features[w]=( w in words)
        
    return features


#print( findfeatures(movie_reviews.words('neg/cv000_29416.txt')))
# bag of words
featuresets = [(findfeatures(rev), category) for (rev, category) in documents]
#making training and testing set
trainingset= featuresets[:1900]
testingset= featuresets[1900:]

clf= nltk.NaiveBayesClassifier.train(trainingset)
print("naive bayes:    ",nltk.classify.accuracy(clf, testingset))

mnbclassifier = SklearnClassifier(MultinomialNB())
mnbclassifier.train(trainingset)
print("mnbclassifier acc;   ", nltk.classify.accuracy(mnbclassifier, testingset))

bernoulliclassifier = SklearnClassifier(BernoulliNB())
bernoulliclassifier.train(trainingset)
print("bernoulliclassifier acc;   ", nltk.classify.accuracy(bernoulliclassifier, testingset))


logistic_classifier = SklearnClassifier(LogisticRegression())
logistic_classifier.train(trainingset)
print("logistic_classifier acc;   ", nltk.classify.accuracy(logistic_classifier, testingset))


votedclf= voteclassifier( clf,mnbclassifier,bernoulliclassifier)

print("votedclf accuracy:" ,nltk.classify.accuracy(votedclf, testingset))

print("classification:  ", votedclf.classify(testingset[1][0]) , "confidence: ", votedclf.confidence(testingset[1][0]))

print("classification:  ", votedclf.classify(testingset[2][0]) , "confidence:  ", votedclf.confidence(testingset[2][0]))
print("classification:  ", votedclf.classify(testingset[3][0]) , "confidence:  ", votedclf.confidence(testingset[3][0]))
print("classification:  ", votedclf.classify(testingset[4][0]) , "confidence:  ", votedclf.confidence(testingset[4][0]))


#pickling to save our object module ince trained
'''classifierf= open("naivebayes.pickle" , "rb")
clf= pickle.load(classifierf)
classifierf.close() '''

clf.show_most_informative_features(15)

'''saveclassifier= open("naivebayes.pickle", "wb")
pickle.dump(clf, saveclassifier)
saveclassifier.close() '''





