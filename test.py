from flask import Flask, render_template, redirect, request
from scraper import Scraper
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import string
import numpy as np
import os
import json
from nltk import word_tokenize
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import svm
import csv
import time
import swifter
import pickle
import json
import googletrans
from googletrans import Translator
from googletrans import LANGUAGES
import httpx
from textblob import TextBlob

start_time = time.time()
# load the data into a DataFrame
df = pd.read_csv("uploads/Dataclean.csv", sep=',', encoding='latin1')

# extract the tweet text and the labels
text = df["Stemming"].tolist()
labels = df["Sentiment"].tolist()
    
# create the transform
vectorizer = TfidfVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# encode document
X = vectorizer.transform(text)
    
# scores = X.toarray()
# print(scores)
    
# feature_names = vectorizer.get_feature_names()

# create dataframe with Tfidf scores
# df = pd.DataFrame(scores, columns=feature_names)

# print dataframe
# print(df)

        
# split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=0)

# train an SVM model on the training data
model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
    
modelrbf = SVC(kernel="rbf")
modelrbf.fit(X_train, y_train)
y_predrbf = modelrbf.predict(X_test)

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
        
f1_score1 = f1_score(y_test, y_pred, average='macro')
accuracy_score1 = accuracy_score(y_test, y_pred)
precision_score1 = precision_score(y_test, y_pred, average='macro')
recall_score1 = recall_score(y_test, y_pred, average='macro')

f1_score2 = f1_score(y_test, y_predrbf, average='macro')
accuracy_score2 = accuracy_score(y_test, y_predrbf)
precision_score2 = precision_score(y_test, y_predrbf, average='macro')
recall_score2 = recall_score(y_test, y_predrbf, average='macro')
    
predictions = np.array(y_test)
ground_truth = np.array(y_pred)
    
predictions1 = np.array(y_test)
ground_truth1 = np.array(y_predrbf)

# Calculate the confusion matrix
confusion_mat_linear = confusion_matrix(ground_truth, predictions)
confusion_mat_rbf = confusion_matrix(ground_truth1, predictions1)
    
print(confusion_mat_linear)
print(confusion_mat_rbf)
    
elapsed_time = time.time() - start_time
elapsed_time_minutes = elapsed_time / 60
print(f'Elapsed Time: {elapsed_time_minutes:.2f} minute')