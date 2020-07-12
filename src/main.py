'''
This script generates the similarity matrix to use
on the flask app
'''
# Load necessary packages
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Load helper functions
from nlp_helpers import 

def main():
    # load up the mongo client and tables
    client = MongoClient('localhost', 27017)
    db = client['podcasts']
    table1 = db['show_listings']
    table2 = db['episode_listings']

    # Load MongoDB info into a pandas df
    podcasts = pd.DataFrame(list(table1.find()))
    episodes = pd.DataFrame(list(table2.find()))



if __name__ == '__main__':
    main()