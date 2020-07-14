'''
This script generates the bag of words dataframe
To run this from command line, run 
python src/bag_of_words_generator.py
'''
# Load necessary packages
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Load helper functions
from nlp_helpers import build_text_vectorizer, bag_of_words

def main():
    # load up the mongo client and tables
    print("[INFO] Loading Database and Tables")
    client = MongoClient('localhost', 27017)
    db = client['podcasts']
    table1 = db['show_listings']
    table2 = db['episode_listings']

    # Load MongoDB info into a pandas df
    print("[INFO] Loading Podcasts and Episodes into Dataframes")
    podcasts = pd.DataFrame(list(table1.find()))
    episodes = pd.DataFrame(list(table2.find()))

    # get bag of words from function
    print("[INFO] Generating a bag of words for each podcast")
    bag_words = bag_of_words(podcasts, episodes)

    # Build text-to-vectorizer
    print("[INFO] Building Vectorizer and Vocabulary")
    vectorizer, vocabulary = build_text_vectorizer(bag_words.bag_of_words,
                                                use_tfidf=True,
                                                use_stemmer=True,
                                                max_features=5000)
    X = vectorizer(bag_words.bag_of_words)

    # Load into a dataframe
    print("[INFO] Loading bag of words into dataframe")
    df_bag = pd.DataFrame(X, index=bag_words.index, columns=vocabulary)

    # Pickle the dataframe, store in data folder
    print("[INFO] Pickling the data")
    df_bag.to_pickle('data/df_bag_of_words.pkl')


if __name__ == '__main__':
    main()