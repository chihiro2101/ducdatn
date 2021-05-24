import math
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
import os
import re
import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('averaged_perceptron_tagger')

def sim_2_sent(df_tfidf):
    matrix_cossim = cosine_similarity(df_tfidf, df_tfidf)
    return matrix_cossim


def sim_with_title(list_sentences_frequency, title):
    simWithTitle = []
    for sent_vector in list_sentences_frequency:
        simT = cosine_similarity([sent_vector], [title])[0][0]
        simWithTitle.append(simT)
    return simWithTitle


def sim_with_doc(list_sentences_frequency, document_vector):
    simWithDoc = []
    for sent_vector in list_sentences_frequency:
        simD = cosine_similarity([sent_vector], [document_vector])[0][0]
        simWithDoc.append(simD)
    return simWithDoc


def count_noun(sentences, option = False):
    if option == False:
        number_of_nouns = [0]*len(sentences)
    else:
        number_of_nouns = []
        for sentence in sentences:
            stopwords = nltk.corpus.stopwords.words('english')
            text_tokens = nltk.word_tokenize(sentence)
            tokens_without_sw = [word for word in text_tokens if not word in stopwords]
            post = nltk.pos_tag(tokens_without_sw)
            noun_list = ['NNP', 'NNPS']
            num = 0
            for k, v in post:
                if v in noun_list:
                    num += 1
            number_of_nouns.append(num)
    return number_of_nouns


def preprocess_raw_sent(raw_sen):
    raw_sent = raw_sen.lower()
    symbols = "!\"#$%&()*+-./:;,\'<=>?@[\]^_`{|}~\n"
    for i in symbols:
        raw_sent = raw_sent.replace(i, '')
    remove_number = "".join((item for item in raw_sent if not item.isdigit())).strip()
    stopwords = nltk.corpus.stopwords.words('english')
    text_tokens = nltk.word_tokenize(remove_number)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    # stemmer= PorterStemmer()
    # stemmed_word = [stemmer.stem(word) for word in tokens_without_sw]

    preprocessed_sent = (" ").join(tokens_without_sw)
    return preprocessed_sent
