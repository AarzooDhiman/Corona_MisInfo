#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:12:47 2020

@author: aarzoo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:11:39 2020

@author: aarzoo
"""

import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import matplotlib.pyplot as plt
from collections import defaultdict
#import nltk 
#.tokenize import word_tokenize
#nltk.download()
import preprocess
#from MulticoreTSNE import MulticoreTSNE as TSNE

train= pd.read_csv(os.getcwd()+"/feb 2020/26-29/combined_csv(26-29).csv", engine='python')
#train=train.head(1000)
print (train.columns.values)
df_text = train['text'].tolist()
data_processed_text = list(map(preprocess.process_text, df_text))
#data_processed_text = [i for i in data_processed_text if i] 
#print (data_processed_text[:10])



train['cleaned']= data_processed_text
train= train.dropna(subset=['cleaned'])
train.to_csv(os.getcwd()+"/feb 2020/26-29/combined_csv(26-29)_cleaned.csv")
print("done preprocessing")