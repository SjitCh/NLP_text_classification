parameters = {
"data_column": 'reason' ,#Column Select, Training Column
"target_column": 'Tag',#Column Select, Target Variable
"train_split": 0.7,# Float, Train-Test Split(0-1), Desc - Relative amount of data to be used for training. Default 1
#"dataObjectName": ,#String, Model name

#"vectorizer_file_path": ,#dataObject, Vectorizer File. Desc - Select a vectorizer if already created, else select parameters for vectorizer below
#"vectorizer_name": ,#string, Vectorizer Name. Desc - If creating a new vectorizer, name of the vectorizer
#"seed": #number, Random number seed, Desc - Default 42

}

import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
from datetime import date
import os,gc
from joblib import dump,load
from sklearn.metrics import confusion_matrix, f1_score,precision_score,recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from joblib import dump,load
print("started")
df = pd.read_csv("/home/jovyan/data/data/output/da49652c-ba7d-4531-b610-a50cf856d841/solve_266/user_3519/473ebe0f-e991-40a4-9b2f-0100823de23d/OUT_1_udf_tenant_72254edf2c647b9353276e82a9d229e881011427_user_3519_final_data_new.csv",sep=";")
df1 = pd.read_csv("/home/jovyan/data/data/output/da49652c-ba7d-4531-b610-a50cf856d841/solve_266/user_3519/177029f1-efcd-47bd-bfe9-921e9e58df29/OUT_1_sql_new.csv",sep=";")
df = df.merge(df1,how='inner',on='TICKET_ID')
df = df[df.Tag!='Other'].append(df[df.Tag=='Other'].sample(n=10000,replace=False),ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=.5).reset_index(drop=True)

data_column = parameters["data_column"]
train_split = parameters.get("train_split", 1)
target_column = parameters["target_column"]
seed = parameters.get("seed", 42)

start = time.time()

input_dataset = df[[data_column,target_column]]
del df
del df1

class TfidfWord2VecVectorizer( BaseEstimator, TransformerMixin ):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=self.analyzer_func)
        tfidf.fit(X)
        self.max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(self.get_max_idf)
        for w, i in tfidf.vocabulary_.items():
            self.word2weight[w] = tfidf.idf_[i]

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    def analyzer_func(self, x):
        return x

    def get_max_idf(self):
        return self.max_idf


def tokenize_data(input_data):
    nlp = spacy.load("en")
    tokenizer = Tokenizer(nlp.vocab)
    string_data = [str(data) for data in input_data]
    tokenized_data = [[str(w) for w in doc] for doc in tokenizer.pipe(string_data, batch_size=50)]
    return tokenized_data
print("loading w2v")
w2v_rsn = load("/home/jovyan/data/data/output/da49652c-ba7d-4531-b610-a50cf856d841/solve_266/user_3519/data/dataobject_W2V_Word_Emb_Reason_2019_08_23.jobli")
print("w2v loaded")
#rename

tag_ids={'T1':0,  'T2':1,  'T3':2,  'T4':3,  'T5':4,  'T6':5,  'T7':6,  'T8':7,  'T9':8,  'T10':9,  'T11':10,  'T12':11,  'T13':12,  'T14':13,  'T15':14,  'T16':15,  'T17':16,  'T18':17,  'T19':18,  'T20':19,  'T21':20,  'T22':21,  'T23':22,  'T24':23,  'T25':24,  'T26':25,  'T27':26,  'T28':27,  'T29':28,  'T30':29,  'T31':30,  'T32':31,  'T33':32,  'T34':33,  'T35':34,  'Other':35}

queues=np.array([0,1,2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35])

input_dataset[target_column]=input_dataset[target_column].replace(tag_ids)

X = input_dataset.drop(target_column, axis=1)
Y = input_dataset[target_column]
if train_split != 1:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_split, random_state=seed)
else:
    X_train, X_test, y_train, y_test = X, None, Y, None
del X
del Y
print("tokenize data")
tokenized_data_train = tokenize_data(X_train[data_column])
tokenized_data_test = tokenize_data(X_test[data_column])
print("creating w2v object")
vectorizer = TfidfWord2VecVectorizer(word2vec=w2v_rsn)
print("fitting w2v")
vectorizer.fit(tokenized_data_train)
print("transform using w2v")
train_data = vectorizer.transform(tokenized_data_train)

del X_train
del tokenized_data_train

gc.collect()

print("declare MLP object")
clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(512,256,128,32), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=seed, tol=0.0001, verbose=True, warm_start=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10),n_jobs=-1)
print('fitting started')

clf.fit(train_data, y_train)

train_predictions = clf.predict(train_data)
train_probabilities = clf.predict_proba(train_data)[:, 1]

train_conf_matrix = pd.DataFrame(confusion_matrix(y_train, train_predictions, labels=list(queues)))
train_conf_matrix.columns = list(queues)
train_conf_matrix.index = list(queues)
train_conf_matrix['Precision'] = precision_score(y_train, train_predictions,labels=list(queues),average=None)
train_conf_matrix['Recall'] = recall_score(y_train, train_predictions,labels=list(queues),average=None)
train_conf_matrix['F1_Score'] = f1_score(y_train, train_predictions,labels=list(queues),average=None)
train_conf_matrix['Ticket_Count']=train_conf_matrix.iloc[:,:36].sum(axis=1)
output = {
        "Train confusion matrix": train_conf_matrix
}

end = time.time()
if X_test is not None:
    test_data = vectorizer.transform(tokenized_data_test)
    del X_test
    del tokenized_data_test
    gc.collect()
    test_predictions = clf.predict(test_data)
    test_probabilities = clf.predict_proba(test_data)[:, 1]
    test_conf_matrix = pd.DataFrame(confusion_matrix(y_test, test_predictions, labels=list(queues)))
    test_conf_matrix.columns = list(queues)
    test_conf_matrix.index = list(queues)
    test_conf_matrix['Precision'] = precision_score(y_test,test_predictions,labels=list(queues),average=None)
    test_conf_matrix['Recall'] = recall_score(y_test,test_predictions,labels=list(queues),average=None)
    test_conf_matrix['F1_Score'] = f1_score(y_test,test_predictions,labels=list(queues),average=None)
    test_conf_matrix['Ticket_Count']=test_conf_matrix.iloc[:,:36].sum(axis=1)
    output["Test confusion matrix"] = test_conf_matrix
else:
    output["Test confusion matrix"] = pd.DataFrame({"Test Confusion Matrix details": [
        "No test confusion matrix plotted as entire data is used for training."]})

output["Total time taken to train"] = pd.DataFrame({"Train time": [f"{(end - start)/3600} hours"]})

output_train = pd.DataFrame(output['Train confusion matrix'])
output_test = pd.DataFrame(output['Test confusion matrix'])
output_time = pd.DataFrame(output['Total time taken to train'])

reverse_mapping = {0:"T1",  1:"T2",  2:"T3",  3:"T4",  4:"T5",  5:"T6",  6:"T7",  7:"T8",  8:"T9",  9:"T10",  10:"T11",  11:"T12",  12:"T13",  13:"T14",  14:"T15",  15:"T16",  16:"T17",  17:"T18",  18:"T19",  19:"T20",  20:"T21",  21:"T22",  22:"T23",  23:"T24",  24:"T25",  25:"T26",  26:"T27",  27:"T28",  28:"T29",  29:"T30",  30:"T31",  31:"T32",  32:"T33",  33:"T34",  34:"T35",  35:"Other"}

output_train.rename(columns=reverse_mapping,index=reverse_mapping,inplace=True)
output_test.rename(columns=reverse_mapping,index=reverse_mapping,inplace=True)

#print(output)
#return output
print("done")

output_test.to_csv("/home/jovyan/data/data/output/da49652c-ba7d-4531-b610-a50cf856d841/solve_266/user_3519/data/MLP_w2v_test_{}.csv".format(date.today()))
output_train.to_csv("/home/jovyan/data/data/output/da49652c-ba7d-4531-b610-a50cf856d841/solve_266/user_3519/data/MLP_w2v_train_{}.csv".format(date.today()))
