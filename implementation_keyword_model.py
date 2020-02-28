import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from joblib import load,dump
from math import gcd

stoplist = set(stopwords.words('english'))

Ticket_reason = 'healow app issue'

tokens = word_tokenize(Ticket_reason)

tokens = [token for token in tokens if token not in stoplist]

Ticket_reason_freq={'n_grams':[],'freq':[]}

for l in range(1,4):
    bgs = nltk.ngrams(tokens,l)
    fdist = nltk.FreqDist(bgs)
    for k,v in fdist.items():
        x=''
        for m in range(l):
            x = x + ' ' +k[m]
        x = x.strip()
        Ticket_reason_freq['n_grams'].append(x)
        Ticket_reason_freq['freq'].append(v)

Ticket_reason_freq=pd.DataFrame(Ticket_reason_freq)
Ticket_reason_freq['Tag']='Tag' ## dummy values
Ticket_reason_freq['Rank']=0 ## dummy values
Ticket_reason_freq=Ticket_reason_freq[list(top_keywords.columns)]

temp_df=pd.DataFrame()

temp_df = top_keywords.append(Ticket_reason_freq)
temp_df = temp_df[temp_df.duplicated('n_grams',keep='last')]
if temp_df.shape[0]!=0:
    temp_df1= temp_df.groupby('Tag').count()[['n_grams']].reset_index()

    if any(temp_df1.duplicated('n_grams')):
        x=list(temp_df1[temp_df1.duplicated('n_grams',keep=False)]['n_grams'])
        if max(temp_df1['n_grams']) in x: ### check for duplicate max values of ngrams
            temp_df2=temp_df[temp_df.Tag.isin(list(temp_df1[temp_df1.n_grams==max(temp_df1['n_grams'])].Tag))].groupby('Tag').sum()[['Rank']].reset_index() ### sum of rank grouped by Tag
            temp_df2 = temp_df2.merge(ticket_count,on='Tag',how='left')
            temp_df2['scaled_Rank']=1.0*temp_df2['Rank']/temp_df2['ticket_count']
            y = list(temp_df2[temp_df2.duplicated('scaled_Rank',keep=False)]['scaled_Rank'])
            if min(temp_df2['scaled_Rank']) in y:
                print(temp_df2.iloc[temp_df2[['ticket_count']].idxmax(axis = 0),0])
            else:
                print(temp_df2.iloc[temp_df2[['scaled_Rank']].idxmin(axis = 0),0])
        else:
            print(temp_df1.iloc[temp_df1[['n_grams']].idxmax(axis = 0),0])
    else:
        print(temp_df1.iloc[temp_df1[['n_grams']].idxmax(axis = 0),0])
else:
    print("No Match")
