import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

# Sample cricket data
cricket_data = [
    "Cricket is a bat-and-ball game played between two teams of eleven players.",
    "The objective is to score runs by hitting the ball with a bat and running between the wickets.",
    "Cricket is widely regarded as the second most popular sport in the world after soccer.",
    "The game originated in England and is now played in various formats, including Test matches, One Day Internationals, and Twenty20.",
    "The International Cricket Council (ICC) is the governing body for the sport.",
    "Famous cricketers include Sachin Tendulkar, Brian Lara, and Sir Don Bradman."
]

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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    if user_input.lower() == "bye":
        return "Cricket ChatBot: Goodbye!"
    else:
        response = generate_response(user_input)
        return "Cricket ChatBot: " + response

if __name__ == '__main__':
    app.run(debug=True)
