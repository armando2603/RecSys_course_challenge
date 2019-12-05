"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects
from MatrixFactorization.IALSRecommender import IALSRecommender
import pandas as pd
from Evaluator.evaluation import evaluate
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from pathlib import Path

data_folder = Path(__file__).parent.absolute()

test = True

Data = DataManager()

if test:
    urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), use_validation_set=False,
                                                            leave_random_out=True)
else:
    urm_train = Data.get_urm()

MyRecommender = UserKNNCFRecommender(urm_train)
MyRecommender.fit(topK=20, shrink=30)

# SecondRecommender = UserKNNCFRecommender(urm_train)
# SecondRecommender.fit(topK=249, shrink=1)

if test:

    evaluate(urm_test, MyRecommender)

else:

    users = Data.get_target_users()

    recommended_list = []
    for user in tqdm(users):
        recommended_items = MyRecommender.recommend(user, 10)
        # if len(recommended_items) == 0:
        #     recommended_items = SecondRecommender.recommend(user, 10)
        items_strings = ' '.join([str(i) for i in recommended_items])
        recommended_list.append(items_strings)

    submission = pd.DataFrame(list(zip(users, recommended_list)), columns=['user_id', 'item_list'])
    submission.to_csv(data_folder / 'my_submission.csv', index=False)

