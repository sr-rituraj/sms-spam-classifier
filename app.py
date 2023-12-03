import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def text_transform(text):

  text=text.lower()  #lowercasing
  text=nltk.word_tokenize(text)  #tokenization
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)   # removing special characters
  
  #stopwords
  z=[]
  for i in y:
    if i in stopwords.words('english'):
      pass
    else:
      z.append(i)

  #Stemming
  stemmer = PorterStemmer()
  stemmed_words = [stemmer.stem(word) for word in z]
  return ' '.join(stemmed_words)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam classifier")
input_sms=st.text_input("Enter the message")

if st.button("Predict"):

  #preprocess
  transformed_sms=text_transform(input_sms)

  #vectorize
  vector_input=tfidf.transform([transformed_sms])

  #predict
  result=model.predict(vector_input)[0]

  #Display
  if result==1:
    st.header("Spam")
  else:
    st.header("Not Spam")

