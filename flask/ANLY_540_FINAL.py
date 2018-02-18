import json
import gensim, operator, sys
from scipy import spatial
import numpy as np
from gensim.models import KeyedVectors, word2vec
import io
import watson_developer_cloud
import pprint
import flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import os

# Reading the json file:

myfile = open('anly540-90.json', 'r')
mydata = myfile.read()

# named entity recognition using NLU :
# classifying text into pre defined categories(text,)

import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1 as NaturalLanguageUnderstanding
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions

natural_language_understanding = NaturalLanguageUnderstanding(
    username="1057ee62-29d6-4810-bb63-fc02f9b3f36f",
    password="YGfWUOdSygSS",
    version="2017-11-20")
response = natural_language_understanding.analyze(
    text=mydata,
    features=Features(entities=EntitiesOptions())
)
# print(json.dumps(response, indent=2))

stopwords = set(nltk.corpus.stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


def tokenize(text):
    # first tokenize by sentence, then by word to ensure that
    # punctuation is caught as it's own token

    tokens = [word for sent in nltk.sent_tokenize(text) \
              for word in tokenizer.tokenize(sent) if word not in stopwords]
    lmtzr = WordNetLemmatizer()

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token) and len(token) > 1:
            filtered_tokens.append(token.lower())

    bigrams = [' '.join(bigram) for bigram in nltk.bigrams(filtered_tokens)]
    trigrams = [' '.join(trigram) for trigram in nltk.trigrams(filtered_tokens)]
    stems = [lmtzr.lemmatize(t, 'v') for t in filtered_tokens]

    for bigram in bigrams:
        if bigram not in stopwords:
            stems.append(bigram)
    for trigram in trigrams:
        if trigram not in stopwords:
            stems.append(trigram)

    return stems


def clstr_lda(num_topics, stories):
    """

    :rtype: object
    """
    n_top_words = 10

    tf_vectorizer = CountVectorizer(max_df=0.8, min_df=0.2, max_features=1000,
                                    tokenizer=tokenize, ngram_range=(1, 3))

    tf = tf_vectorizer.fit_transform(stories)

    lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=200,
                                    learning_method='online', learning_offset=9.,
                                    random_state=1)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # print("\nLDA Topics:")
    # print top topic words
    # topics = dict()
    # for topic_idx, topic in enumerate(lda.components_):
    # topics[topic_idx] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    # print("Topic #%d:" % topic_idx)
    # print(" | ".join([tf_feature_names[i]
    # for i in topic.argsort()[:-n_top_words - 1:-1]]))
    # print()
    # return topics


myfile = open('anly540-90.json', 'r')
mydata = json.loads(myfile.read())

texts = list()
for post in mydata['posts']:
    if post['text']:
        texts.append(post['text'])

topics = clstr_lda(8, texts)

# myfile =  open('anly540-90.json','r')
# mydata = json.loads(myfile.read())

# texts = list()
# for post in mydata['posts']:
#   if post['text']:
#      texts.append(post['text'])




vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(texts)

true_k = 3  # number of clusters

model = KMeans(n_clusters=true_k,
               init='k-means++', max_iter=100,
               n_init=1)

model.fit(X)

# print("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
# print("Cluster %d:" % i),
# for ind in order_centroids[i, :10]:
# print(' %s' % terms[ind]),
# print


# print("\n")
# print("Prediction")

topic_taxonomy = {
    "offers":
        {
            "free": "shipping trial product",
            "deal": "discount countdown limited ",
            "unlimited": "stock warranty buy"
        },
    "companies":
        {
            "amazon": "shopping order department",
            "facebook": "likes social advertisement",
            "google": "maps search doodle"
        },
    "returns":
        {
            "shipped": "free checkout prime",
            "cashback": "credit cash days",
            "balance": "card money transfer"
        }
}


# model_path =  "/Users/apoorva/Desktop/python/PycharmProjects/ANLY 540-90/assignment 7/"

def load_word2vec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(modelFile, binary=flagBin, limit=50000)
    print('Finished loading ' + modelName + ' model...')
    return model


model_word2vec = load_word2vec_model('word2Vec', 'GoogleNews-vectors-negative300.bin', True)


def vocab_check(vectors, words):
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())

    return output


# function calculating similarity between two strings :
def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))
    if len(s1words) > 0 and len(s2words) > 0:
        return vectors.n_similarity(s1words, s2words)
    else:
        return 0


myfile = open('anly540-90.json', 'r')
mydata = json.loads(myfile.read())

titles = set()
for post in mydata['posts']:
    if post['title']:
        titles.add(post['title'])

titles = list(titles)

for i in range(700):
    target = titles[i]
    scores = []
    subtitles = titles.copy()
    subtitles.pop(i)
    for title in subtitles:
        scores.append((title, calc_similarity(target, title, model_word2vec)))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # print('Target Title: ', target)
    # print('----------------------------------------------------------')
    # for j in range(10):
    # print(sorted_scores[j][0])

    # print('\n\n')


# function checks whether the input words are present in the vocabulary for the model
def vocab_check(vectors, words):
    output = list()
    for word in words:
        if word in vectors.vocab:
            output.append(word.strip())

    return output


def calc_similarity(input1, input2, vectors):
    s1words = set(vocab_check(vectors, input1.split()))
    s2words = set(vocab_check(vectors, input2.split()))

    output = vectors.n_similarity(s1words, s2words)
    return output


# function takes an input string, runs similarity for each item in topic_taxonomy, sorts and returns top 3 results
def classify_topics(input, vectors):
    feed_score = dict()
    for key, value in topic_taxonomy.items():
        max_value_score = dict()
        for label, keywords in value.items():
            max_value_score[label] = 0
            topic = (key + ' ' + keywords).strip()
            max_value_score[label] += float(calc_similarity(input, topic, vectors))

        sorted_max_score = sorted(max_value_score.items(), key=operator.itemgetter(1), reverse=True)[0]
        feed_score[sorted_max_score[0]] = sorted_max_score[1]
    return sorted(feed_score.items(), key=operator.itemgetter(1), reverse=True)[:3]





    # App config.


app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    title = TextField('Title:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def topic_classifier():
    form = ReusableForm(request.form)

    print (form.errors)
    if request.method == 'POST':
        title = request.form['title']

        results = classify_topics(title, model_word2vec)
        if form.validate():
            # Save the comment here.
            flash(json.dumps(results, indent=4)

                  )
        else:
            flash('All the form fields are required. ')

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
