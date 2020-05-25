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
from nltk.tokenize import word_tokenize
import preprocess
#from MulticoreTSNE import MulticoreTSNE as TSNE
import os

os.chdir("/home/aarzoo/Aarzoo/code/covid/feb2020/21-25")
train= pd.read_csv('/home/aarzoo/Aarzoo/code/covid/feb2020/21-25/combined_csv(21-25)_cleaned.csv', engine='c')
#train=train.head(1000)

#df_text = train['text'].tolist()
#data_processed_text = list(map(preprocess.process_text, df_text))
#data_processed_text = [i for i in data_processed_text if i] 
#print (data_processed_text[:10])



#train['cleaned']= data_processed_text

train= train.dropna(subset=['cleaned'])
#train.to_csv('/home/aarzoo/Aarzoo/code/covid/feb2020/combined_csv(01-5feb)_cleaned.csv')
print("done preprocessing")

'''
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content_train = []
j=0
for em in train['text'].values:
    all_content_train.append(LabeledSentence1(em,[j]))
    j+=1
print("Number of texts processed: ", j)

d2v_model = Doc2Vec(all_content_train, size = 100, window = 10, min_count = 500, workers=7, dm = 1,alpha=0.025, min_alpha=0.001)
d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)

'''

tagged_data = [gensim.models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train['cleaned'].tolist())]
max_epochs = 50
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  

#model = Doc2Vec.load("/home/aarzoo/Aarzoo/code/covid/d2v.model")
model.build_vocab(tagged_data)
#model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

train['embedding'] = list(model.docvecs.vectors_docs)
train.drop(['text'], axis=1)
train.to_csv("combined_cleaned_embed.csv",index=False)

y= model.docvecs.doctag_syn0

kmeans_model = KMeans(n_clusters=50, init='k-means++', max_iter=100) 
X = kmeans_model.fit(model.docvecs.doctag_syn0)

cluster_map = pd.DataFrame()
cluster_map['data_index'] =train.index #['id']
cluster_map['cluster'] = kmeans_model.labels_
cluster_map.to_csv("cluster_map.csv")
'''
from sklearn.cluster import Birch
 
k=4
brc = Birch(branching_factor=50, n_clusters=k, threshold=0.1, compute_labels=True)
brc.fit(d2v_model.docvecs.doctag_syn0)
 
l = brc.predict(d2v_model.docvecs.doctag_syn0)
 
labels = brc.labels_.tolist()
 
 
print ("Clusters: ")
print (l)
'''
plt.bar(range(len(set(kmeans_model.labels_))), np.bincount(kmeans_model.labels_))

plt.ylabel('population')
plt.xlabel('cluster label')
plt.title('population sizes with {} clusters'.format(50))
plt.savefig("clusters_pop.png", dpi=200, bbox_inch='tight')

mean_th = np.mean(np.bincount(kmeans_model.labels_))

#cand_labels=np.bincount(kmeans_model.labels_).argsort()[-5:][::-1]

cand_labels= np.where(np.bincount(kmeans_model.labels_) > mean_th)

cand_clus = defaultdict(list) 
cand_ids = []

def cluster_sample(orig_text, model, idx, preview=15):
    """
    Helper function to display original bio for
    those users modeled in cluster `idx`.
    """
    for idx in np.nditer(cand_labels):
        #print (type(idx))
        # print (np.where(model.labels_ == idx)[0])
        for i,ids in enumerate(np.where(model.labels_ == idx)[0]):
            #print (type(i))
            #print (type(ids))
            cand_ids.append(ids)
            cand_clus[int(idx)].append(orig_text[ids])
            #print(orig_text[ids].replace('\n',' '))
            #print()
            '''if i > preview:
                print('( >>> truncated preview <<< )')
                break'''
        
interest_idx = 5

cluster_sample(train['cleaned'].tolist(), kmeans_model, interest_idx)

#rel = pd.DataFrame.from_dict(cand_clus)
rel = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in cand_clus.items() ]))
rel.to_csv("clusters.csv",index=False)


'''
 
#labels=kmeans_model.labels_.tolist()
#l = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)
pca = PCA(n_components=2).fit(d2v_model.docvecs.doctag_syn0)
datapoint = pca.transform(d2v_model.docvecs.doctag_syn0)
import matplotlib.pyplot as plt



plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()'''