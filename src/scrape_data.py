# Import Necessary Packages
import spotipy
import spotipy.util as util

import requests
import json

import pandas as pd
import numpy as np

from pymongo import MongoClient

import time

class Podcasts():
    '''
    Using the podcast_search method, the class will 
    search for podcast shows in that category.

    Then for each show, find episodes and then store those
    episodes in a MongoDB.

    Inputs
    '''
    def __init__(self, client_id, client_secret, username, redirect_uri='http://localhost:8880/'):
        self.client_id = str(client_id) # client ID generated from spotify 'Spotify for Developers'
        self.client_secret = str(client_secret) # client secret generated from spotify 'Spotify for Developers'
        self.username = username # spotify username
        self.redirect_uri = redirect_uri
        self.token = self._get_token() # token to make requests with
        self.epsiode_table = self._get_episode_table() # MongoDB table to store episodes
        self.podcast_table = self._get_podcast_table() # MongoDB table to store podcasts

    def _get_episode_table(self):
        '''
        Generates table to store the epsiodes of podcasts
        '''
        client = MongoClient('localhost', 27017)
        db = client['podcast_test']
        table = db['episodes_2']

        return table

    def _get_podcast_table(self):
        '''
        Generates table to store the podcasts information
        '''
        client = MongoClient('localhost', 27017)
        db = client['podcast_test']
        table = db['podcasts_2']

        return table

    def _get_token(self):
        '''
        Uses spotipy to generate a token that will be used for requests
        '''
        token = util.prompt_for_user_token(username=self.username,  
                                   client_id=self.client_id,   
                                   client_secret=self.client_secret,     
                                   redirect_uri=self.redirect_uri)

        return token 

    def _is_entry_new(self, entry, table):
        '''
        Checks the table to ensure that the entry 
        is new. Returns True if new, False if not new.
        Inputs
        -----
        entry: string, id of the podcast or episode
        table: mongodb table of podcast or episode
        '''

        if len(list(table.find({'id': str(entry['id'])}))) > 0:
            return False
        else:
            return True



    def podcast_search(self, search):
        '''
        Performs search of similar podcasts
        '''
        # Print a statement
        print('[INFO] Starting search for {}!'.format(search))
        # search endpoint
        endpoint_url = "https://api.spotify.com/v1/search?"

        # get token for use
        token = self._get_token()

        # Initialize lists to store responses
        id_list = []
        name_list = []
        desc_list = []
        explicit_list = []
        images_list = []
        languages_list = []
        publisher_list = []

        type_ = 'show'
        market = 'US'
        limit = 50
        offset = 0

        more_runs = 1
        counter = 0

        # Perform the search
        while((offset <= 1950) & (counter <= more_runs)):
            
            query = f'{endpoint_url}'
            query += f'&q={search}'
            query += f'&type={type_}'
            query += f'&offset={offset}'
            query += f'&market={market}'
            query += f'&limit={limit}'
            
            # send a GET request using query
            response = requests.get(query,
                                headers={"Content-Type":"application/json",
                                        "Authorization":f"Bearer {token}"})
            # load into JSON
            json_response = response.json()
            
            for i in range(len(json_response['shows']['items'])):
                
                id_list.append(json_response['shows']['items'][i]['id'])
                name_list.append(json_response['shows']['items'][i]['name'])
                desc_list.append(json_response['shows']['items'][i]['description'])
                explicit_list.append(json_response['shows']['items'][i]['explicit'])
                images_list.append(json_response['shows']['items'][i]['images'])
                languages_list.append(json_response['shows']['items'][i]['languages'])
                publisher_list.append(json_response['shows']['items'][i]['publisher'])
            
            # determine how many more runs of 50 are required
            more_runs = (json_response['shows']['total'] // limit)
            
            # increase counter by 1
            counter += 1
            
            # increase offset by limit
            offset += limit

        # Initialize dataframe
        podcasts = pd.DataFrame()

        # Store lists into dataframe
        podcasts['id'] = id_list
        podcasts['name'] = name_list
        podcasts['desc'] = desc_list
        podcasts['explicit'] = explicit_list
        podcasts['images'] = images_list
        podcasts['languages'] = languages_list
        podcasts['publisher'] = publisher_list

        # print success statement
        print('[INFO] Successfully Performed Search!')

        # Turn the dataframe into a dictionary
        podcasts_dict = podcasts.to_dict("records")

        # Check to make sure the entries are new
        for each_entry in podcasts_dict:
            new = self._is_entry_new(each_entry, self.podcast_table)

            if new:
                self.podcast_table.insert_one(each_entry)

        # send the podcasts dataframe to seperate method for getting
        # episodes of each podcast
        self._get_episodes(podcasts)

    def _get_episodes(self, podcasts):
        '''
        Takes in pandas dataframe of podcasts based
        off the search in podcast_search

        Goes through each show and gets the episodes.

        Stores into table that was generated from _get_table() method.
        '''
        # Print a status statement
        print('[INFO] Starting search for episodes in each show')

        # Get a list of all the shows from the podcasts dataframe
        ids = podcasts['id']
        names = podcasts['name']

        # parameters for searching
        market = 'US'
        limit = 50
        offset = 0

        token = self._get_token()

        # Initialize lists to store responses
        id_list = []
        podcast_name = []
        episode_name_list = []
        date_list = []
        dur_list = []
        desc_list = []
        audio_preview_url_list = []
        language_list = []

        for id_, name_ in zip(ids, names):
            counter = 0
            more_runs = 1

            # declare endpoint URL
            endpoint_url = f'https://api.spotify.com/v1/shows/{id_}/episodes?'

            while (counter <= more_runs):
                    query = f'{endpoint_url}'
                    query += f'&offset={offset}'
                    query += f'&market={market}'
                    query += f'&limit={limit}'

                    try:
                        response = requests.get(query,
                                                headers = {"Content-Type":"application/json",
                                                        "Authorization":f"Bearer {token}"})
                    except:
                        # wait 60 seconds then try again
                        time.sleep(60)
                        token = self._get_token()
                        response = requests.get(query,
                                                headers = {"Content-Type":"application/json",
                                                        "Authorization":f"Bearer {token}"})
                    # load json response
                    json_response = response.json()
                    
                    # loop through the response and append to the lists
                    for i in range(len(json_response['items'])):
                        id_list.append(json_response['items'][i]['id'])
                        podcast_name.append(name_)
                        episode_name_list.append(json_response['items'][i]['name'])
                        date_list.append(json_response['items'][i]['release_date'])
                        dur_list.append(json_response['items'][i]['duration_ms'])
                        desc_list.append(json_response['items'][i]['description'])
                        audio_preview_url_list.append(json_response['items'][i]['audio_preview_url'])
                        language_list.append(json_response['items'][i]['language'])
                        
                    more_runs = (json_response['total'] // limit)
                    print(more_runs)
                    
                    counter += 1
                    
                    offset += limit

                    # wait 1 second before next call
                    time.sleep(1)

        # Initialize dataframe
        episodes = pd.DataFrame()
            
        episodes['id'] = id_list
        episodes['podcast_name'] = podcast_name
        episodes['episode_name'] = episode_name_list
        episodes['date'] = date_list
        episodes['duration(ms)'] = dur_list
        episodes['description'] = desc_list
        episodes['audio_preview_url'] = audio_preview_url_list
        episodes['language'] = language_list     
