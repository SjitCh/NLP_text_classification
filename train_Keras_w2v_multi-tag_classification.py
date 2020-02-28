##call all parameters such that we can change them later just once.
parameters = {
"data_column": 'reason' ,#Column Select, Training Column
"target_column": 'Tag',#Column Select, Target Variable
"train_split": 0.7,# Float, Train-Test Split(0-1), Desc - Relative amount of data to be used for training. Default 1
#"dataObjectName": ,#String, Model name

#"vectorizer_file_path": ,#dataObject, Vectorizer File. Desc - Select a vectorizer if already created, else select parameters for vectorizer below
#"vectorizer_name": ,#string, Vectorizer Name. Desc - If creating a new vectorizer, name of the vectorizer
#"seed": #number, Random number seed, Desc - Default 42
"hidden_layer_sizes":[512,256,128,64],
"activation":"relu",
"num_classes":28,
"loss":"categorical_crossentropy",
"optimizer":"adam",
"metrics":"adam",
"monitor":"val_loss",
"min_delta":0.0001,
"epochs":500,
"batch_size":64,
"validation_split":0.1,
"patience":5,
"restore_best_weights":True
}

import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
import time
import os,gc
from joblib import dump,load
from sklearn.metrics import confusion_matrix, f1_score,precision_score,recall_score
import keras
from keras.models import Sequential

### create class for converting word to a vector using word2vec embeddings.
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

hidden_layer_sizes=parameters["hidden_layer_sizes"]
activation=parameters["relu"]
num_classes=parameters["num_classes"]
restore_best_weights=parameters["restore_best_weights"]

## Read Dataset
input_dataset = pd.read_csv("")
X = input_dataset.drop(target_column, axis=1)
Y = input_dataset[target_column]
encoder = LabelEncoder()
encoded = encoder.fit(Y)
encoded_y=encoder.transform(Y)
Y=encoded_y
if train_split != 1:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_split, random_state=seed)
else:
    X_train, X_test, y_train, y_test = X, None, Y, None

del X
del Y

y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_test_binary = keras.utils.to_categorical(y_test, num_classes)

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

model = Sequential()
for i in hidden_layer_sizes:
    model.add(Dense(i,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=loss,optimizer=optimizer,metrics=metrics)

callbacks=keras.callbacks.callbacks.EarlyStopping(monitor=monitor,min_delta=min_delta,patience=patience, restore_best_weights=restore_best_weights)

model.fit(train_data, y_train_binary, batch_size = batch_size, epochs = epochs, validation_split=validation_split,callbacks=callbacks)

if X_test is not None:
    test_data = vectorizer.transform(tokenized_data_test)
    del X_test
    del tokenized_data_test
    gc.collect()
    test_predictions = np.argmax(model.predict(test_data),axis =1)
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
