'''
This script provides helper functions for reading
and analyzing the contents of podcast descriptions
'''

# Import Necessary Packages
import numpy as np
import pandas as pd 

from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF 

def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):
    '''
    Build and return a callable for transforming text documents to vectors, as well as
    a vocabulary to map document-vector indicies to words from the corpus. The vectorizer
    will be trained from the text documents in the 'contents' arguments.
    If use_tfidf is True, then the vectorizer will use the TF-IDF algorithm, otherwise
    a Bag-of-Words vectorizer will be used.
    The text will be tokenized be words, and each word will be stemmed iff 'use_stemmer'
    is True.
    If max_features is NOT None, then teh vocabulary will be limited to the max_features
    most common words in the corpus.
    '''