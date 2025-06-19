import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Load and prepare data
with open('Ecommerce_FAQ_Chatbot_dataset.json', 'r') as f:
    data = json.load(f)
faq_list = data['questions'] if isinstance(data, dict) else data
df = pd.DataFrame(faq_list)
df = df.rename(columns={'question': 'Question', 'answer': 'Answer'})

# Setup vectorizer and transform questions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_q = vectorizer.fit_transform(df['Question'])

# Chatbot function
def get_answer(query, threshold=0.2):
    q_tfidf = vectorizer.transform([query])
    sims = cosine_similarity(q_tfidf, tfidf_q)
    idx = np.argmax(sims)
    if sims[0, idx] < threshold:
        return "Sorry, I didn't understand your question. Can you rephrase?"
    return df.at[idx, 'Answer']

# Streamlit UI
st.title("E-Commerce FAQ Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    answer = get_answer(user_input)
    st.write("Answer:", answer)