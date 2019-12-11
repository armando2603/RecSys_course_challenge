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
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
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
from Hybrid.HybridColdRecommender import HybridColdRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Hybrid.HybridZeroRecommender import HybridZeroRecommender
from Hybrid.HybridWarmRecommender import HybridWarmRecommender


data_folder = Path(__file__).parent.absolute()

test = True
threshold = 5
temperature = 'normal'
Data = DataManager()


if test:
    urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), threshold=threshold, temperature=temperature)
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


zero_recommender = None
cold_recommender = None
warm_recommender = None
normal_recommender = None


# MyRecommender = IALSRecommender(urm_train)
# MyRecommender.fit(alpha=6, epochs=20, reg=0.1528993352584987, num_factors=260)
# zero_recommender = UserKNNCBFRecommender(urm_train, ucm_age)
# zero_recommender.fit(shrink=0, topK=250)

# MyRecommender = SLIM_BPR_Cython(urm_train)
# MyRecommender.fit(epochs=198, lambda_i=0.0926694015, lambda_j=0.001697250, learning_rate=0.002391)

if temperature == 'cold' or test==False:
    recommender = IALSRecommender(urm_train)
    recommender.fit(epochs=15, alpha=6.0, num_factors=250.0, reg=0.3)
    cold_recommender = recommender

if temperature == 'zero' or test==False:
    ucm_age, ucm_region, ucm_all = Data.get_ucm()
    zero_recommender = HybridZeroRecommender(urm_train, ucm_all)
    zero_recommender.fit(alpha=0.02, beta=0.9)
    zero_recommender = recommender

if temperature == 'warm' or test==False:
    recommender = ItemKNNCFRecommender(urm_train)
    recommender.fit(topK=20, shrink=30)
    warm_recommender= recommender

if temperature == 'normal':
    # normal_recommender = ItemKNNCFRecommender(urm_train)
    # normal_recommender.fit(topK=20, shrink=30)
    # recommender = SLIMElasticNetRecommender(urm_train)
    # recommender.fit()
    # recommender1 = ItemKNNCFRecommender(urm_train)
    # recommender1.fit(topK=20, shrink=30)
    icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
    recommender = ItemKNNCBFRecommender(urm_train, icm_all)
    recommender.fit(topK=100, shrink=0, feature_weighting='TF-IDF')
    # recommender = ItemKNNSimilarityHybridRecommender(urm_train, recommender1.W_sparse, recommender2.W_sparse)
    # recommender.fit(alpha=0.99, topK=200)

    normal_recommender = recommender

if test:

    if temperature == 'cold':
        result, str_result = evaluator_test.evaluateRecommender(cold_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))

    if temperature == 'zero':
        result, str_result = evaluator_test.evaluateRecommender(zero_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))

    if temperature == 'warm':
        result, str_result = evaluator_test.evaluateRecommender(warm_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))

    if temperature == 'normal':
        result, str_result = evaluator_test.evaluateRecommender(normal_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))
        # res_test = evaluate(urm_test, normal_recommender)
        # print(res_test)

else:

    users = Data.get_target_users()

    recommended_list = []
    for user in tqdm(users):
        if Data.get_cold_users(0)[user]:
            recommended_items = zero_recommender.recommend(user, 10)
        else:
            if Data.get_cold_users(threshold=threshold)[user]:
                recommended_items = cold_recommender.recommend(user, 10)
            else:
                recommended_items = warm_recommender.recommend(user, 10)
        items_strings = ' '.join([str(i) for i in recommended_items])
        recommended_list.append(items_strings)

    submission = pd.DataFrame(list(zip(users, recommended_list)), columns=['user_id', 'item_list'])
    submission.to_csv(data_folder / 'Data/Submissions/After_Debug_submission.csv', index=False)

