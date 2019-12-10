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
from collections import defaultdict
from random import randint, random, uniform

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

        # Build the URM matrix

        users = self.get_raw_users()
        items = self.get_raw_items()
        length = items.shape[0]
        urm = sps.coo_matrix((np.ones(length), (users, items)))
        self.urm = urm.tocsr()
        self.cold_user_list = {}

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
        return self.urm

    def get_ucm(self):
        print('Building UCM ...')

        n_users = self.urm.shape[0]

        # Build the age UCM

        age_df = pd.read_csv('Data/data_UCM_age.csv')

        list_user = list(age_df['row'])
        list_age = list(age_df['col'])

        n_age = max(list_age)+1
        ucm_shape = (n_users, n_age)

        ones = np.ones(len(list_user))

        ucm_age = sps.coo_matrix((ones, (list_user, list_age)), shape=ucm_shape)
        ucm_age = ucm_age.tocsr()

        # Build the Region UCM

        region_df = pd.read_csv('Data/data_UCM_region.csv')

        list_user = list(region_df['row'])
        list_region = list(region_df['col'])

        n_region = max(list_region)+1
        ucm_shape = (n_users, n_region)

        ones = np.ones(len(list_region))

        ucm_region = sps.coo_matrix((ones, (list_user, list_region)), shape=ucm_shape)
        ucm_region = ucm_region.tocsr()

        ucm_all = sps.hstack((ucm_age, ucm_region))
        ucm_all = ucm_all.tocsr()

        return ucm_age, ucm_region, ucm_all

    def get_icm(self):
        urm = self.urm
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

    def get_cold_users(self, threshold=10):
        if not self.cold_user_list:
            urm = self.urm
            n_users, n_items = urm.shape
            cold_user_list = {}
            for user in range(n_users):
                cold_user_list[user] = False

            for user_id in range(n_users):
                start_user_position = urm.indptr[user_id]
                end_user_position = urm.indptr[user_id + 1]

                user_profile = urm.indices[start_user_position:end_user_position]

                if len(user_profile) <= threshold:
                    cold_user_list[user_id] = True

            self.cold_user_list = cold_user_list

        return self.cold_user_list

    def get_warm_users(self, threshold=10):
        urm = self.urm
        n_users, n_items = urm.shape
        warm_user_list = []

        for user_id in range(n_users):
            start_user_position = urm.indptr[user_id]
            end_user_position = urm.indptr[user_id + 1]

            user_profile = urm.indices[start_user_position:end_user_position]

            if len(user_profile) > threshold:
                warm_user_list.append(user_id)

        return warm_user_list


