import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

ps = PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  
nltk.download('stopwords')


def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())



def transform_text(text):
    text = text.lower()
    text = simple_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"⚠️ Error loading model/vectorizer: {e}")
    st.stop()


st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 Email / SMS Spam Classifier")
st.write("This tool classifies text messages as **Spam** or **Not Spam** using a trained ML model.")

input_sms = st.text_area("Enter your message:", placeholder="Type your message here...")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before prediction.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.error("🚨 **Spam Message Detected!**")
        else:
            st.success("✅ **This message is Not Spam.**")

