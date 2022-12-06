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

    nlp=spacy.load("en_core_web_sm")
    # Compute Similarity
    token_1=nlp(first_sentence)
    token_2=nlp(second_sentence)

    similarity_score=token_1.similarity(token_2)


    if (float(similarity_score) > 0.7):
        output_sim = "similar"
    else:
        output_sim = "not similar"
    return output_sim


# def sentiment(sentence_sentiment):
#   # Importing the pipeline function from the transformers
#   from transformers import pipeline
#   # Creating a TextClassificationPipeline for Sentiment Analysis
#   pipe = pipeline(task='sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
#   # Analyzing sentiment
#   pipe(sentence_sentiment)
#   return pipe(sentence_sentiment)


def sentiment(sentence_sentiment):
    from spacytextblob.spacytextblob import SpacyTextBlob
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(sentence_sentiment)
    doc._.blob.polarity
    if (float(doc._.blob.polarity) > 0):
        sentiment = "POSITIVE"
    else:
        sentiment = "NEGATIVE"
    return sentiment


app = Flask(__name__)
app.debug = True
config = None

CORS(app, support_credentials=True)
logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )



@app.route('/sentiment/<string:name1>/')
def hello1(name1):
    sentiment1 = sentiment(name1)
    return sentiment1


@app.route("/similarity/<string:name3>/<string:name4>/")
def hello3(name3, name4):
    similarity1 = text_similarity_2(name3, name4)
    return similarity1



if __name__ == '__main__':  
    app.run(host= '127.0.0.1', port=8090)