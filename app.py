
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

inps = open('classifier.pkl', 'rb')
model = pickle.load(inps)

p = open('cv.pkl', 'rb')
modelcv = pickle.load(p)


@st.cache()



def predict_input(text_input):
    encoded_feelings = [0,1,2,3,4,5]

    Feelings = ["Anger","Fear","Joy","Love","Sadness","Surprise"]

    Feelings_dict = {key : value for key,value in zip(encoded_feelings,Feelings)}
    Input = modelcv.transform(text_input)
    input_prediction = model.predict(Input)
    predict_df = pd.DataFrame(input_prediction.toarray())
    for i in range(0,6):
        if (predict_df.iloc[0:1,i] == 1).item() ==True:
            feeling = Feelings_dict[i]
    
    return feeling





def main():
    html_temp = """ 
    <div style ="background-color:black;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Streamlit Sentiment Prediction ML App</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True) 
    text = st.text_input('Put in Sentence')
    results = ''

    if st.button('Predict'):
        results = predict_input([text])
        st.success(results)

if __name__=='__main__': 
    main()





