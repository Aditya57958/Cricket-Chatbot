#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import numpy as np
import pandas as pd
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample cricket data
cricket_data = pd.read_csv('')

# Preprocessing
nltk.download("punkt")
nltk.download("wordnet")
stemmer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.lemmatize(token) for token in tokens if token not in string.punctuation]
    return " ".join(tokens)

preprocessed_data = [preprocess(sentence) for sentence in cricket_data]

# Generate response
def generate_response(user_input):
    preprocessed_input = preprocess(user_input)
    preprocessed_data.append(preprocessed_input)

    TfidfVec = TfidfVectorizer()
    tfidf = TfidfVec.fit_transform(preprocessed_data)
    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    index = np.argmax(similarity_scores)

    return cricket_data[index]

# Main loop
print("Cricket ChatBot: Hello! I'm a Cricket ChatBot. Ask me anything about cricket.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "bye":
        print("Cricket ChatBot: Goodbye!")
        break
    else:
        response = generate_response(user_input)
        print("Cricket ChatBot:", response)

