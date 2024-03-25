import sklearn
import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    # Lower case:
    text = text.lower()
    # Word Tokenization:
    text = nltk.word_tokenize(text)

    # Removing special characters:
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Stopwords and punctuations:

    text = y[:]  # cloning of list
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming:

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/sms Spam Classifier")

input_sms = st.text_input("Enter the message: ")

if st.button('predict'):

## Steps:
## Step1: Preprocess (Using transform_text() function)

    transformed_sms = transform_text(input_sms)

## Step2: Vectorize

    vector_input = tfidf.transform([transformed_sms])

## Step3: Predict

    result = model.predict(vector_input)[0]

## Step4: Display

    if result == 1:
      st.header("Spam")
    else:
      st.header("Not Spam")
