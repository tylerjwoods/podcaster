'''
This script provides a class called PodcastRecommender
which is a content based recommender based on the
description of podcast episodes.

This will be used within the web app.
'''
# Import Necessary Packages
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class PodcastRecommenderApp():
    '''
    Content based item recommender
    '''
    def __init__(self):
        # Read pickle of the similarity matrix
        self.similarity_df = pd.read_pickle('data/similarity_matrix.pkl')
        self.item_names = self.similarity_df.index
        self.similarity_measure = cosine_similarity

        # Read pickle of the bag of words
        self.item_counts = pd.read_pickle('data/df_bag_of_words.pkl')
    
    def get_recommendations(self, item, n=5):
        '''
        Returns the top n items related to the item passed in
        INPUT
        ------
            item - STRING - spotify_id of podcast in the original dataframe
            n    - INT - Number of top related items to return
        OUTPUT
        ------
            spotify_id's - list of the top n related podcast

        For a given podcast find the n most most similar items to it (this can be done
        using the similarity matrix created in the fit method)
        '''
        return self.item_names[self.similarity_df.loc[item].values.argsort()[-(n+1):-1]].values[::-1]

    def get_user_profile(self, items):
        '''
        Takes a list of podcasts and returns a user profile. A vector representing the likes
        of the user.
        INPUT
        -----
            items - LIST - list of podcast serials user likes 
        OUTPUT
        -----
            user_profile - np.array - array representing the likes of the user
                The columns of this will match the columns of the trained on matrix
            
        Using the list of items liked by the user, create a profile which will be a 1 x number of features array.
        This should be the addition of the values for all liked item features.
        '''
        user_profile = np.zeros(self.item_counts.shape[1]) 
        for item in items:
            user_profile += self.item_counts.loc[item].values  
        
        return user_profile
        
    def get_user_recommendation(self, items, n=5):
        '''
        Takes a list of podcasts user liked and returns the top n items for that user

        INPUT
        ------
            items - LIST - list of podcasts user likes / has seen
            n - INT - number of items to return

        OUTPUT
        ------
            items - LIST - n recommended podcasts

        Makes use of the get_user_profile method to create a user profile that will be used to get
        the similarity to all items and recommend the top n.
        '''
        num_items = len(items)
        user_profile = self.get_user_profile(items)

        user_sim = self.similarity_measure(self.item_counts, user_profile.reshape(1, -1))

        return self.item_names[user_sim[:,0].argsort()[-n(num_items+n):-num_items]].values[::-1]