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
from pathlib import Path

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
        print('Building ICM from items...')

        # 1 - Price

        item_data = pd.read_csv(data_folder / "data_ICM_price.csv")
        price = item_data.reindex(columns=['row', 'data'])  # sto trattando i prezzi degli item
        price.sort_values(by='row', inplace=True)  # this seems not useful, values are already ordered
        price_list = [[a] for a in price['data']]
        icm_price = MultiLabelBinarizer(sparse_output=True).fit_transform(price_list)
        icm_price_csr = icm_price.tocsr()

        # 2 - Asset

        item_data = pd.read_csv(data_folder / "data_ICM_asset.csv")
        asset = item_data.reindex(columns=['row', 'data'])  # sto trattando i prezzi degli item
        asset.sort_values(by='row', inplace=True)  # this seems not useful, values are already ordered
        asset_list = [[a] for a in asset['data']]
        icm_asset = MultiLabelBinarizer(sparse_output=True).fit_transform(asset_list)
        icm_asset_csr = icm_asset.tocsr()

        icm = sps.hstack((icm_price_csr, icm_asset_csr))
        return icm.tocsr()


