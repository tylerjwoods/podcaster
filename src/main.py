'''
This script generates the similarity matrix to use
on the flask app
'''
# Load necessary packages
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Load helper functions
from nlp_helpers import build_text_vectorizer, bag_of_words
from podcast_recommender import PodcastRecommender

def main():
    # load up the mongo client and tables
    client = MongoClient('localhost', 27017)
    db = client['podcasts']
    table1 = db['show_listings']
    table2 = db['episode_listings']

    # Load MongoDB info into a pandas df
    podcasts = pd.DataFrame(list(table1.find()))
    episodes = pd.DataFrame(list(table2.find()))

    # get bag of words from function
    bag_words = bag_of_words(podcasts, episodes)

    # Build text-to-vectorizer
    vectorizer, vocabulary = build_text_vectorizer(bag_words.bag_of_words,
                                                use_tfidf=True,
                                                use_stemmer=True,
                                                max_features=5000)
    X = vectorizer(bag_words.bag_of_words)

    # Load into a dataframe
    df_bag = pd.DataFrame(X, index=bag_words.index, columns=vocabulary)

    # Create a PodcastRecommender class
    rec = PodcastRecommender()

    # Fit the df_bag on the Recommender
    rec.fit(df_bag)



if __name__ == '__main__':
    main()