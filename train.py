import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import get_dummies
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, plot_confusion_matrix


df_train = pd.read_csv(filepath_or_buffer='train.txt', sep = ';', names=['text','sentiment'])
df_test = pd.read_csv(filepath_or_buffer='test.txt', sep = ';', names=['text','sentiment'])
df_valid = pd.read_csv(filepath_or_buffer='val.txt', sep = ';', names=['text','sentiment'])

def cleaning(text):
    clean = ''.join(u for u in text if u not in (',','!','.','?',':',';','"','-','\''))
    return clean

df_train['text'] = df_train['text'].apply(cleaning)
df_test['text'] = df_test['text'].apply(cleaning)

def separation(data):
    y = data['sentiment']
    x = data['text']
    return x,y

x_train, y_train = separation(df_train)
x_test, y_test = separation(df_test)

#Text to numbers

y_train = get_dummies(y_train)
y_test = get_dummies(y_test)

modelcv = CountVectorizer(stop_words='english')

x_train = modelcv.fit_transform(x_train)
x_test = modelcv.transform(x_test)

tokens = modelcv.get_feature_names()

df1 = pd.DataFrame(data=x_train.toarray(), columns=tokens)

model = LabelPowerset(LogisticRegression(max_iter=160000))
model.fit(x_train, y_train)

predict = model.predict(x_test)

print(accuracy_score(y_test, predict)*100)

predict_df = pd.DataFrame(predict.toarray())

encoded_feelings = predict_df.columns.to_list()

Feelings = ["Anger","Fear","Joy","Love","Sadness","Surprise"]

Feelings_dict = {key : value for key,value in zip(encoded_feelings,Feelings)}

def predict_input(text_input):
    Input = modelcv.transform(text_input)
    input_prediction = model.predict(Input)
    predict_df = pd.DataFrame(input_prediction.toarray())
    for i in range(0,6):
        if (predict_df.iloc[0:1,i] == 1).item() ==True:
            feeling = Feelings_dict[i]
    
    return feeling

