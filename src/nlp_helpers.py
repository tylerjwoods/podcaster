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

    Example to call this:
    vectorizer, vocabulary = build_text_vectorizer(contents,
                                              use_tfidf=True,
                                              use_stemmer=True,
                                              max_features=5000)
    X = vectorizer(contents)
    '''
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary

def bag_of_words(podcasts, episodes):
    '''
    Takes in podcasts and episodes dataframes and combines
    the description of each episode into on column to form
    a bag of words.
    Inputs
    ------
    podcasts: pandas dataframe; with columns:
                _id,
                spotify_id
                name
                desc
                explicit
                images
                languages
                publisher
    episodes: pandas dataframe; with columns:
                '_id', 
                'spotify_id', 
                'podcast_name', 
                'episode_name', 
                'date',
                'duration(ms)', 
                'description', 
                'audio_preview_url', 
                'language'
    Returns
    -------
    df: pandas dataframe with index as the name of the podcast and 
        column bag_of_words
    '''
    # create new pandas dataframe with the spotify_id from podcasts
    ### CHANGE TO NAME INSTEAD OF spotify_id ###
    df = podcasts[['name']].copy()
    # create column placeholder 
    df['bag_of_words'] = ""

    # iterate over each row in podcasts
    for index, row in podcasts.iterrows():
        words = ''
        # add the desc of the podcast to words
        words += row['desc'] + ' '
        # iterate over each row in episodes for that podcast
        for each_episode in episodes[episodes['podcast_name']==row['name']]['description']:
            words += each_episode + ' '
        # assign the bag of words to df
        df.loc[index, 'bag_of_words'] = words 
        
    # set the index as spotify_id
    df.set_index('name', inplace=True)
    return df