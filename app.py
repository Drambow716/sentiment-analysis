
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import emotions
import random

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    <h1 style ="color:white;text-align:center;">Understand your feelings with AI</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.subheader("Describe your feelings in general") 
    text = st.text_input('Please add your general feelings')
    st.subheader("What do you think about life ?") 
    text1 = st.text_input('Please be honest')
    st.subheader("Do you like your work ?") 
    text2 = st.text_input("We promise we won't tell your boss")
    results = ''

    choices = [text,text1,text2]
    choice = random.choice(choices)

    if st.button('Understand your feelings'):
        results = predict_input([choice.lower()])
        if results == "Anger":
            st.warning("You have anger issues 🤬")
            with st.expander("Understand your Feelings in depth"):
                st.write(emotions.Anger_defintion)
                st.subheader("How to Takcle Anger Issues ?")
                st.write(emotions.Anger_text)
                st.image("https://cdn.pixabay.com/photo/2020/02/08/18/24/sunset-4830931_960_720.jpg")
        if results == "Fear":
            st.warning("You have deep fear issues 😱")
            with st.expander("Understand your Feelings in depth"):
                st.write(emotions.fear_defition)
                st.image("https://www.hunted.com/blog/wp-content/uploads/2019/01/How-to-Overcome-Fear.jpg")
                st.subheader("Here are Ten ways to fight your fears")
                st.write(emotions.fear_text)
                
        if results == "Joy":
            st.success("You are a Happy motherfucker 🤗")
            with st.expander("Tips on how stay happy forever"):
                st.image("data:image/jpeg;base64,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")
                st.write(emotions.joy_text)

        if results == "Love":
            st.success("You are in Love ❤️‍🔥")
            with st.expander("Understand your Feelings in depth"):
                st.video("https://www.youtube.com/watch?v=HEXWRTEbj1I&t=4s")

        if results == "Sadness":
            st.warning("You are a sad Person 😭")
            with st.expander("Understand your Feelings in depth"):
                st.write(emotions.sadness_definition)
                st.image("https://media-cldnry.s-nbcnews.com/image/upload/t_social_share_1200x630_center,f_auto,q_auto:best/newscms/2019_42/1495563/sadness-inside-out-today-main-tease-191018.jpg")
                st.subheader("How Can You Deal With Sad Feelings?")
                st.write(emotions.sadness_text)
                

        if results == "Surprise":
            st.warning("You have mixed feelings together with Surprise 😲")
            with st.expander("Want to know more about Surprise ?"):
                st.image("https://c.tenor.com/6BHseki8RSYAAAAM/dexter-erik-king.gif")
                st.write(emotions.surprise_definition)        


        

if __name__=='__main__': 
    main()





