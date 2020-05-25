#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:21:24 2020

@author: aarzoo
"""

from langdetect import detect
#from nltk.tokenize.casual import TweetTokenizer
import string
from collections import Counter
#from nltk.util import everygrams
import re

def get_features(texts):
    if type(texts) is str:
        texts = [texts]
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return sess.run(embed(texts))

def remove_stopwords(stop_words, tokens):
    res = []
    for token in tokens:
        if not token in stop_words:
            res.append(token)
    return res

def process_text(text):
    #print (text)
    try:
      if (detect(text)=='en'):
        text = text.encode('ascii', errors='ignore').decode()
        text = text.lower()
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'#+', ' ', text )
        #text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
        text = re.sub(r"([A-Za-z]+)'s", r"\1 is", text)
        #text = re.sub(r"\'s", " ", text)     text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"won't", "will not ", text)
        text = re.sub(r"isn't", "is not ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub('\W', ' ', text)
        #text = re.sub(r'\d+', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip()
        return text
    except:
        return ("empty")

def lemmatize(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_list = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token, 'v')
        if lemma == token:
            lemma = lemmatizer.lemmatize(token)
        lemma_list.append(lemma)
    # return [ lemmatizer.lemmatize(token, 'v') for token in tokens ]     return lemma_list


def process_all(text):
    text = process_text(text)
    return ' '.join(remove_stopwords(stop_words, text.split()))