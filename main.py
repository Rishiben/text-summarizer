from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import numpy as np
import networkx as nx
from heapq import nlargest
import webbrowser


def preprocess_text(text):
    """
    Function to preprocess the input text by removing stop words and tokenizing sentences.
    """
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    clean_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word not in stop_words and word.isalnum()]
        clean_sentences.append(' '.join(words))
    return clean_sentences


app = Flask(__name__, static_folder='static')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """     
    Function to generate a summary of the input text using TextRank algorithm. 
    """
    text = request.form['text']
    size = int(request.form['size'])

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stop words from each sentence
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if word.lower() not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    # Calculate the weighted frequencies of each sentence
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / max_frequency)

    # Calculate the scores for each sentence based on its words' frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]

    # Get the top N sentences with the highest scores
    summary_sentences = nlargest(size, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    return render_template('summary.html', summary=summary)

#opening localhost with port 5000 in browser
link = "http://127.0.0.1:5000"
webbrowser.open_new_tab(link)

#run main app
app.run()

