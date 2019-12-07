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
from sklearn.preprocessing import MultiLabelBinarizer, normalize

"""
This class is used to create dataframes from the data
We create 3 matrices:
    - URM: User Rating Matrix
    - ICM: Item Content Matrix
    - UCM: User Content Matrix
"""

data_folder = Path(__file__).parent.parent.absolute()


class DataManager(object):

    def __init__(self):

        self.train = pd.read_csv(data_folder / 'Data/data_train.csv')
        #self.items = pd.read_csv('Data/items.csv')

    def get_target_users(self):
        sample_df = pd.read_csv(data_folder / 'Data/alg_sample_submission.csv')
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

    def get_icm(self):
        urm = self.get_urm()
        n_item = urm.shape[1]
        print('Building ICM from items...')

        # 1 - Price

        price_df = pd.read_csv(data_folder / "Data/data_ICM_price.csv")
        list_price = list(price_df['data'])
        list_item = list(price_df['row'])
        n_price = len(list(price_df['data'].unique()))
        icm_shape = (n_item, n_price)

        ones = np.ones(len(list_price))
        icm_price = sps.coo_matrix((ones, (list_item, list_price)), shape=icm_shape)
        icm_price = icm_price.tocsr()


        # 2 - Asset

        asset_df = pd.read_csv(data_folder / "Data/data_ICM_asset.csv")
        list_asset = list(asset_df['data'])
        list_item = list(asset_df['row'])
        n_asset = len(list(asset_df['data'].unique()))
        icm_shape = (n_item, n_asset)

        ones = np.ones(len(list_asset))
        icm_asset = sps.coo_matrix((ones, (list_item, list_asset)), shape=icm_shape)
        icm_asset = icm_asset.tocsr()

        # icm = sps.hstack((icm_price, icm_asset))
        # return icm.tocsr()
        return icm_price, icm_asset


