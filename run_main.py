"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
import numpy as np
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
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from Hybrid.Hybrid2PredRecommender import Hybrid2PredRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Hybrid.HybridCBFRecommender import HybridCBFRecommender

data_folder = Path(__file__).parent.absolute()

test = True

Data = DataManager()



if test:
    urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), threshold=5, cold=True)
    #urm_train, urm_test = Data.split_warm_leave_one_out_random()
    # urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, use_validation_set=False, leave_random_out=True)
    # urm_train, urm_test = train_test_holdout(Data.get_urm(), train_perc=0.8)
    # urm_train, urm_valid = train_test_holdout(urm_train, train_perc=0.8)
    # evaluator_validation = EvaluatorHoldout(urm_valid, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
else:
    urm_train = Data.get_urm()

# earlystopping_keywargs = {"validation_every_n": 5,
#                               "stop_on_validation": True,
#                               "evaluator_object": evaluator_validation,
#                               "lower_validations_allowed": 5,
#                               "validation_metric": "MAP"
#                           }


# icm_price, icm_asset = Data.get_icm()
ucm_age, ucm_region, ucm_all = Data.get_ucm()

cold_recommender = Hybrid2PredRecommender(urm_train, ucm_all)
cold_recommender.fit(alpha=0.6, beta=0.000002, gamma=0.000002)
# MyRecommender = IALSRecommender(urm_train)
# MyRecommender.fit(alpha=6, epochs=20, reg=0.1528993352584987, num_factors=260)
# cold_recommender = HybridCBFRecommender(urm_train, ucm_age, ucm_region)
# cold_recommender.fit(alpha=0.2, beta=0.15)
# cold_recommender = UserKNNCBFRecommender(urm_train, ucm_all)
# cold_recommender.fit(shrink=0, topK=400)
#cold_recommender = ItemKNNCFRecommender(urm_train)
#cold_recommender.fit(topK=20, shrink=30)
# MyRecommender = SLIM_BPR_Cython(urm_train)
# MyRecommender.fit(epochs=198, lambda_i=0.0926694015, lambda_j=0.001697250, learning_rate=0.002391)
# cold_recommender = TopPop(urm_train)
# cold_recommender.fit()

if test:

     # evaluate(urm_test, MyRecommender)
    result, str_result = evaluator_test.evaluateRecommender(cold_recommender)
    print(result)

else:

    users = Data.get_target_users()

    recommended_list = []
    for user in tqdm(users):
        # recommended_items = MyRecommender.recommend(user, 10)
        if Data.get_cold_users(threshold=5)[user]:
            recommended_items = cold_recommender.recommend(user, 10)
        else:
            recommended_items = warm_recommender.recommend(user, 10)
        items_strings = ' '.join([str(i) for i in recommended_items])
        recommended_list.append(items_strings)

    submission = pd.DataFrame(list(zip(users, recommended_list)), columns=['user_id', 'item_list'])
    submission.to_csv(data_folder / 'Data/warm_and_cold_submission.csv', index=False)

