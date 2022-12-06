from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import threading
import logging
from flask.logging import default_handler

import spacy
from rake_nltk import Rake

from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import numpy as np
import itertools

import bs4 as bs
import urllib.request
import re
import nltk




def text_similarity_2(first_sentence, second_sentence):
    from sentence_transformers import SentenceTransformer

    sentence_1 = (
        first_sentence  # "Three years later, the coffin was still full of Jello."
    )
    sentence_2 = second_sentence  # "The person box was packed with jelly many dozens of months later."
    sentences = []
    sentences.append(sentence_1)
    sentences.append(sentence_2)

    model = SentenceTransformer("bert-base-nli-mean-tokens")
    sentence_embeddings = model.encode(sentences)
    from sklearn.metrics.pairwise import cosine_similarity

    text_similarity = cosine_similarity(
        [sentence_embeddings[0]], [sentence_embeddings[1]]
    )
    # type(text_similarity)
    if float(text_similarity[0]) > 0.7:
        output_sim = "similar"
    else:
        output_sim = "not similar"
    return output_sim

    # return str(float(text_similarity[0]))


def sentiment(sentence_sentiment):
    # Importing the pipeline function from the transformers
    from transformers import pipeline

    # Creating a TextClassificationPipeline for Sentiment Analysis
    pipe = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    # Analyzing sentiment
    pipe(sentence_sentiment)
    # print(type(pipe(sentence_sentiment)))
    return pipe(sentence_sentiment)[0]["label"]




app = Flask(__name__)
CORS(app)






@app.route("/sentiment/<string:name1>/")
# @cross_origin()
def hello1(name1):
    sentiment1 = sentiment(name1)
    return sentiment1


@app.route("/similarity/<string:name3>/<string:name4>/")
# @cross_origin()
def hello3(name3, name4):
    similarity1 = text_similarity_2(name3, name4)
    return similarity1





if __name__ == "__main__":
    app.run(host="0.0.0.0")
    # app.run()
