#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:01:46 2020

@author: aarzoo
"""

import statistics
import seaborn as sns
import pandas as pd
import os

filepath = "/home/aarzoo/Aarzoo/code/CAA-NRC/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Affect-Intensity-Lexicon/NRC-AffectIntensity-Lexicon.txt"
os.chdir("/home/aarzoo/Aarzoo/code/covid/feb2020/21-25")
df= pd.read_csv("/home/aarzoo/Aarzoo/code/covid/feb2020/21-25/combined_csv(21-25)_cleaned.csv")

emolex_df = pd.read_csv(filepath,  sep= '\t')

emo_anger = emolex_df[emolex_df['AffectDimension']=='anger']
emo_sad = emolex_df[emolex_df['AffectDimension']=='sadness']
emo_joy = emolex_df[emolex_df['AffectDimension']=='joy']
emo_fear = emolex_df[emolex_df['AffectDimension']=='fear']

data_processed_text =df['cleaned'].tolist()
#term score emo

emo_list = []
emo_names = ['anger_list','sad_list','fear_list','joy_list']


print(emo_anger.head(5))

emo_anger= emo_anger.set_index('term')
emo_anger = emo_anger.transpose()
anger = emo_anger.to_dict()

a_score_full = []

count= 0

for text in data_processed_text:
    if (count%10000==0):
      print (count)
    a_score = []
    for key in anger.keys():
        try:
            if key in text.split(' '):
                #print (key)
                #print (text)
                a_score.append(anger[key]['score'])
        except:
            pass
    if len(a_score)>0:
        score= statistics.mean(a_score)
    else:
        score = 0
    a_score_full.append(score)
    count+=1


df['anger'] = a_score_full

print ("anger done")
df.to_csv("emo_dat.csv")

print(emo_sad.head(5))

emo_sad= emo_sad.set_index('term')
emo_sad = emo_sad.transpose()
anger = emo_sad.to_dict()

a_score_full = []

count= 0

for text in data_processed_text:
    if (count%10==0):
      print (count)
    a_score = []
    for key in anger.keys():
        try:
            if key in text.split(' '):
                #print (key)
                #print (text)
                a_score.append(anger[key]['score'])
        except:
            pass
    if len(a_score)>0:
        score= statistics.mean(a_score)
    else:
        score = 0
    a_score_full.append(score)
    count+=1


df['sadness'] = a_score_full

print ("sad done")
df.to_csv("emo_dat.csv")
print(emo_fear.head(5))

emo_fear= emo_fear.set_index('term')
emo_fear = emo_fear.transpose()
anger = emo_fear.to_dict()

a_score_full = []

count= 0

for text in data_processed_text:
    if (count%100==0):
      print (count)
    a_score = []
    for key in anger.keys():
        try:
            if key in text.split(' '):
                #print (key)
                #print (text)
                a_score.append(anger[key]['score'])
        except:
            pass
    if len(a_score)>0:
        score= statistics.mean(a_score)
    else:
        score = 0
    a_score_full.append(score)
    count+=1


df['fear'] = a_score_full

print ("fear done")
df.to_csv("emo_dat.csv")
print(emo_joy.head(5))

emo_joy= emo_joy.set_index('term')
emo_joy = emo_joy.transpose()
anger = emo_joy.to_dict()

a_score_full = []

count= 0

for text in data_processed_text:
    if (count%10==0):
      print (count)
    a_score = []
    for key in anger.keys():
        try:
            if key in text.split(' '):
                #print (key)
                #print (text)
                a_score.append(anger[key]['score'])
        except:
            pass
    if len(a_score)>0:
        score= statistics.mean(a_score)
    else:
        score = 0
    a_score_full.append(score)
    count+=1


df['joy'] = a_score_full

print ("joy done")



    
df.to_csv("emo_dat.csv")

print ("saved to file")

d1= list(filter((0.0).__ne__, df['anger'].tolist()))
d2= list(filter((0.0).__ne__, df['sadness'].tolist()))
d3= list(filter((0.0).__ne__, df['fear'].tolist()))
d4= list(filter((0.0).__ne__, df['joy'].tolist()))

plot_list =[]

for d in d1:
    plot_list.append(('anger',d))
for d in d2:
    plot_list.append(('sadness',d))
for d in d3:
    plot_list.append(('fear',d))
for d in d4:
    plot_list.append(('joy',d))
    
plot_df = pd.DataFrame(plot_list)
plot_df=plot_df.rename(columns={0: "Emotion", 1: "Emotion Score"})
plot_df.to_csv("plot_df.csv")
#sns.catplot(x="Sentiment Category", y="Sentiment Score", kind="violin", inner=None, data=df2)
ax = sns.catplot( x ='Emotion', y='Emotion Score', kind="violin", inner=None, data=plot_df)
#sns.swarmplot(x="Emotion", y="Emotion Score", color="k", data=plot_df, ax=g.ax);
ax.savefig("emo.png", dpi=200, bbox_inch='tight')

#dfp['fear'].astype(bool).sum(axis=0)
print ("sadness")
print (df[df["sadness"] >0.5].count())
print ("fear")
print (df[df["fear"] >0.5].count())
print ("anger")
print (df[df["anger"] >0.5].count())
print ("joy")
print (df[df["joy"] >0.5].count())
