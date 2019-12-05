"""
Created on 01/10/19

@author: Giuseppe Serna
"""
import numpy as np
import scipy.sparse as sps
import pandas as pd
import sklearn as sk
from sklearn import feature_extraction
from pathlib import Path

"""
This class is used to create dataframes from the data
We create 3 matrices:
    - URM: User Rating Matrix
    - ICM: Item Content Matrix
    - UCM: User Content Matrix
"""

data_folder = Path("Data/")


class DataManager(object):

    def __init__(self):

        self.train = pd.read_csv(data_folder / 'data_train.csv')
        #self.items = pd.read_csv('Data/items.csv')

    def get_target_users(self):
        sample_df = pd.read_csv(data_folder / 'alg_sample_submission.csv')
        target_users = sample_df['user_id']
        return target_users

    def get_raw_users(self):
        return self.train['row']

    def get_raw_items(self):
        return self.train['col']

    def get_users(self):
        users = self.train['row'].unique()
        return np.sort(users)

    def get_items(self):
        items = self.train['col'].unique()
        return np.sort(items)

    def get_urm(self):
        print('Building URM...')
        users = self.get_raw_users()
        items = self.get_raw_items()
        length = items.shape[0]
        urm = sps.coo_matrix((np.ones(length), (users, items)))
        urm = urm.tocsr()
        return urm

    def get_ucm(self, urm):
        print('Building UCM from URM...')

        ucm_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(urm.T)
        ucm_tfidf = ucm_tfidf.T
        return ucm_tfidf

    '''def get_icm(self):
        print('Building ICM from items...')

        # 1 - Artists

        # In this case data is order by artist
        #artist_df = self.items.reindex(columns=[['track_id'], ['artist_id']])
        #artist_df.sort_values(by='track_id', inplace=True)

        artists = self.items['artist_id']
        items = self.items['track_id']
        ones = np.ones(len(items))
        urm_shape = self.get_urm().shape

        icm_artist = sps.coo_matrix((ones, (items, artists)), shape=urm_shape)
        icm_artist_csr = icm_artist.tocsr()

        # 2 - Albums

        albums = self.items['album_id']
        items = self.items['track_id']
        ones = np.ones(len(items))
        urm_shape = self.get_urm().shape

        icm_albums = sps.coo_matrix((ones, (items, albums)), shape=urm_shape)
        icm_albums_csr = icm_albums.tocsr()

        # 3 - Duration

        duration = self.items['duration_sec']
        items = self.items['track_id']
        ones = np.ones(len(items))
        urm_shape = self.get_urm().shape

        icm_duration = sps.coo_matrix((ones, (items, duration)), shape=urm_shape)
        icm_duration_csr = icm_albums.tocsr()

        # 4 stack together

        icm = sps.hstack((icm_artist_csr, icm_albums_csr, icm_duration_csr))
        return icm.tocsr()

    '''

