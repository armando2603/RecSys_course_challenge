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
import numpy as np
import scipy.sparse as sps
from DataManager.IncrementalSparseMatrix import IncrementalSparseMatrix
import math
from tqdm import tqdm

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
        return self.urm.tocsr().copy()

    def get_ucm(self):
        print('Building UCM ...')

        n_users = self.urm.shape[0]

        # Build the age UCM

        age_df = pd.read_csv(data_folder /'Data/data_UCM_age.csv')

        list_user = list(age_df['row'])
        list_age = list(age_df['col'])

        n_age = max(list_age)+1
        ucm_shape = (n_users, n_age)

        ones = np.ones(len(list_user))

        ucm_age = sps.coo_matrix((ones, (list_user, list_age)), shape=ucm_shape)
        ucm_age = ucm_age.tocsr()

        # Build the Region UCM

        region_df = pd.read_csv(data_folder /'Data/data_UCM_region.csv')

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

        # price_df = pd.read_csv(data_folder / "Data/data_ICM_price.csv")
        # list_price = list(price_df['data'])
        # list_item = list(price_df['row'])
        #
        # intervallo1 = np.arange(1001, step=200)
        # intervallo2 = np.arange(start=2000, stop=10001, step=1000)
        # intervallo3 = np.arange(start=20000, stop=100001, step=10000)
        # intervallo = np.append(intervallo1, intervallo2)
        # intervallo = np.append(intervallo, intervallo3)
        # list_price = np.array(list_price)
        # list_price = list_price * 100000
        # list_price = np.digitize(list_price, intervallo)
        #
        # n_price = max(list_price)+1
        #
        # icm_shape = (n_item, n_price)
        #
        # ones = np.ones(len(list_price))
        # icm_price = sps.coo_matrix((ones, (list_item, list_price)), shape=icm_shape)
        # icm_price = icm_price.tocsr()
        icm_price = self.get_icm_price(200)

        # 2 - Asset

        # asset_df = pd.read_csv(data_folder / "Data/data_ICM_asset.csv")
        # list_asset = list(asset_df['data'])
        # list_item = list(asset_df['row'])
        #
        # interval1 = np.arange(1001, step=1000 / 3)
        # interval2 = np.arange(start=1500, stop=2001, step=500)
        # interval3 = np.arange(start=3000, stop=10001, step=1000)
        # interval4 = np.array([100000])
        # final_interval = np.append(interval1, interval2)
        # final_interval = np.append(final_interval, interval3)
        # final_interval = np.append(final_interval, interval4)
        # list_asset = np.array(list_asset)
        # list_asset = list_asset * 100000
        # list_asset = np.digitize(list_asset, final_interval)
        # n_asset = max(list_asset)+1
        # icm_shape = (n_item, n_asset)
        #
        # ones = np.ones(len(list_asset))
        # icm_asset = sps.coo_matrix((ones, (list_item, list_asset)), shape=icm_shape)
        # icm_asset = icm_asset.tocsr()

        # icm_all = sps.hstack((icm_price, icm_asset))
        # icm_all.tocsr()

        icm_asset = self.get_icm_asset(200)

        # 3 - Sub_class

        sub_df = pd.read_csv(data_folder / "Data/data_ICM_sub_class.csv")
        list_sub = list(sub_df['col'])
        list_item = list(sub_df['row'])
        n_sub = max(sub_df['col'].unique())

        icm_shape = (n_item, n_sub + 1)

        ones = np.ones(len(list_sub))
        icm_sub = sps.coo_matrix((ones, (list_item, list_sub)), shape=icm_shape)
        icm_sub = icm_sub.tocsr()


        icm_all = sps.hstack((icm_price, icm_asset, icm_sub))
        icm_all.tocsr()

        return icm_price, icm_asset, icm_sub, icm_all

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


    def modify_urm_item_warm(self, urm, thresold=10):
        warm_items_mask = np.ediff1d(urm.tocsc().indptr) > thresold
        warm_items = np.arange(urm.shape[1])[warm_items_mask]

        urm = urm[:, warm_items]

        return urm

    def modify_urm_user_warm(self, urm, thresold=1):
        warm_users_mask = np.ediff1d(urm.tocsr().indptr) > thresold
        warm_users = np.arange(urm.shape[0])[warm_users_mask]

        urm = urm[warm_users, :]
        return urm

    def get_urm_warm_items(self, threshold=1):
        users = self.get_raw_users()
        items = self.get_raw_items()
        length = items.shape[0]

        urm_csc = self.urm.tocsc()

        item_interactions = np.ediff1d(urm_csc.indptr)
        warm_items = item_interactions > threshold
        new_users = []
        new_items = []
        for index in np.arange(length):
            if warm_items[items[index]]:
                new_users.append(users[index])
                new_items.append((items[index]))

        new_length = len(new_items)

        urm = sps.coo_matrix((np.ones(new_length), (new_users, new_items)))
        urm = urm.tocsr()

        return urm

    def get_urm_warm_users(self, threshold=1):

        n_users, n_items = self.urm.shape

        urm_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                    auto_create_col_mapper=False, n_cols=n_items)

        for user_id in range(n_users):
            start_user_position = self.urm.indptr[user_id]
            end_user_position = self.urm.indptr[user_id + 1]

            user_profile = self.urm.indices[start_user_position:end_user_position]

            if len(user_profile) > threshold:
                user_interaction_items_train = user_profile
                user_interaction_data_train = self.urm.data[start_user_position:end_user_position]

                urm_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

        warm_urm = urm_train_builder.get_SparseMatrix()
        warm_urm = sps.csr_matrix(warm_urm)
        user_no_item_train = np.sum(np.ediff1d(warm_urm.indptr) == 0)

        if user_no_item_train != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train,
                                                                                  user_no_item_train / n_users * 100,
                                                                                  n_users))
        return warm_urm

    def get_urm_warm_users_items(self, threshold_user=10, threshold_item=10):

        # Elimino Items

        users = self.get_raw_users()
        items = self.get_raw_items()
        length = items.shape[0]

        urm_csc = self.urm.tocsc()

        item_interactions = np.ediff1d(urm_csc.indptr)
        warm_items = item_interactions > threshold_item
        new_users = []
        new_items = []
        for index in np.arange(length):
            if warm_items[items[index]]:
                new_users.append(users[index])
                new_items.append((items[index]))

        new_length = len(new_items)

        urm = sps.coo_matrix((np.ones(new_length), (new_users, new_items)))
        urm = urm.tocsr()

        #### Elimino Users

        n_users, n_items = urm.shape

        urm_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                    auto_create_col_mapper=False, n_cols=n_items)

        for user_id in range(n_users):
            start_user_position = urm.indptr[user_id]
            end_user_position = urm.indptr[user_id + 1]

            user_profile = urm.indices[start_user_position:end_user_position]

            if len(user_profile) > threshold_user:
                user_interaction_items_train = user_profile
                user_interaction_data_train = urm.data[start_user_position:end_user_position]

                urm_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

        warm_urm = urm_train_builder.get_SparseMatrix()
        warm_urm = sps.csr_matrix(warm_urm)
        user_no_item_train = np.sum(np.ediff1d(warm_urm.indptr) == 0)

        if user_no_item_train != 0:
            print("Warning: {} ({:.2f} %) of {} users have no Train items".format(user_no_item_train,
                                                                                  user_no_item_train / n_users * 100,
                                                                                  n_users))

        return warm_urm


    def create_weight_age_matrix(self):
        age_df = pd.read_csv('Data/data_UCM_age.csv')
        list_user = np.array(age_df['row'])
        list_age = np.array(age_df['col'])

        n_user = len(list_user)

        shape = self.urm.shape[0]

        weight_matrix_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=shape,
                                                    auto_create_col_mapper=False, n_cols=shape)

        for index_1 in tqdm(np.arange(n_user)):

            user_1 = list_user[index_1]

            list_weight = np.zeros(n_user)
            for index_2 in np.arange(n_user):
                list_weight[index_2] = abs(list_age[index_1] - list_age[index_2])
                # weight = self.compute_age_similarity(list_age[index_1], list_age[index_2])

            list_weight = list_weight / 10
            list_weight = np.negative(list_weight)
            list_weight = np.exp(list_weight)

        weight_matrix_builder.add_data_lists([user_1] * len(list_user), list_user, list_weight)

        weight_matrix = weight_matrix_builder.get_SparseMatrix()

        weight_matrix = sps.csr_matrix(weight_matrix)

        return weight_matrix

    def get_icm_price(self, n_clusters):
        """
        Create an ICM (tracks, n_clusters) gathering tracks according to their duration in milliseconds.
        Each cluster has the same number of tracks.
        :param n_clusters: number of clusters
        :return: icm: the item content matrix
        """

        urm = self.get_urm()

        tracks_df = pd.read_csv('Data/data_ICM_price.csv')
        prices = tracks_df['data'].values

        dur_idx = np.argsort(prices)[::-1]

        item_per_cluster = int(urm.shape[1] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), 'ICM prices:'):

            # # If last iteration
            if i == n_clusters - 1:
                items = dur_idx[item_per_cluster * i:]
            else:
                items = dur_idx[item_per_cluster * i: item_per_cluster * (i + 1)]

            rows.extend(items)
            cols.extend([i for x in range(len(items))])
            data.extend([1 for x in range(len(items))])

        icm = sps.csr_matrix((data, (rows, cols)), shape=(urm.shape[1], n_clusters))

        return icm

    def get_icm_asset(self, n_clusters):
        """
        Create an ICM (tracks, n_clusters) gathering tracks according to their duration in milliseconds.
        Each cluster has the same number of tracks.
        :param n_clusters: number of clusters
        :return: icm: the item content matrix
        """

        urm = self.get_urm()

        asset_df = pd.read_csv('Data/data_ICM_asset.csv')
        assets = asset_df['data'].values

        dur_idx = np.argsort(assets)[::-1]

        item_per_cluster = int(urm.shape[1] / n_clusters)

        rows = []
        cols = []
        data = []

        for i in tqdm(range(n_clusters), 'ICM asset:'):

            # # If last iteration
            if i == n_clusters - 1:
                items = dur_idx[item_per_cluster * i:]
            else:
                items = dur_idx[item_per_cluster * i: item_per_cluster * (i + 1)]

            rows.extend(items)
            cols.extend([i for x in range(len(items))])
            data.extend([1 for x in range(len(items))])

        icm = sps.csr_matrix((data, (rows, cols)), shape=(urm.shape[1], n_clusters))

        return icm
