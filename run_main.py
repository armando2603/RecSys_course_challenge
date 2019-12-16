"""
Created on 01/10/19

@author: Giuseppe Serna
"""
from tqdm import tqdm
import matplotlib.pyplot as pyplot
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
import scipy.sparse
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from pathlib import Path
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Notebooks_utils.data_splitter import train_test_holdout
from Hybrid.HybridColdRecommender import HybridColdRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Hybrid.HybridZeroRecommender import HybridZeroRecommender
from Hybrid.HybridWarmRecommender import HybridWarmRecommender
from KNN.UserSimilarityHybridRecommender import UserSimilarityHybridRecommender


data_folder = Path(__file__).parent.absolute()

test = True
threshold = 10
temperature = 'normal'
Data = DataManager()
urm_train = Data.get_urm()

valid = True


# sparse_matrix = scipy.sparse.load_npz('Data/csr_matrix_age.npz')

if test:
    urm_train, urm_test = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature=temperature)
    #urm_train, urm_test = Data.split_warm_leave_one_out_random()
    # urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, use_validation_set=False, leave_random_out=True)
    # urm_train, urm_test = train_test_holdout(Data.get_urm(), train_perc=0.8)
    # urm_train, urm_valid = train_test_holdout(urm_train, train_perc=0.8)
    if valid:
        urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=threshold, temperature='valid')
        evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])
else:
    urm_train = Data.get_urm()

# earlystopping_keywargs = {"validation_every_n": 5,
#                               "stop_on_validation": True,
#                               "evaluator_object": evaluator_valid,
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

if temperature == 'cold' or test is False:


    recommender = HybridWarmRecommender(urm_train)
    recommender.fit(alpha=0.95)



    cold_recommender = recommender

if temperature == 'zero' or test is False:
    # icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
    ucm_age, ucm_region, ucm_all = Data.get_ucm()
    recommender = HybridZeroRecommender(urm_train, ucm_all)
    recommender.fit(alpha=0.02, beta=0.9)


    zero_recommender = recommender

if temperature == 'warm' or test is False:



    recommender = ItemKNNCFRecommender(urm_train)
    recommender.fit(shrink=30, topK=20)



    warm_recommender= recommender

if temperature == 'normal' and test is True:
    # icm_price, icm_asset, icm_sub, icm_all = Data.get_icm()
    # # ucm_age, ucm_region, ucm_all = Data.get_ucm()

    # x_tick = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
    # MAP_per_k_test = []
    #
    # MAP_per_k_valid = []

    # for alpha in x_tick:
    #     recommender.fit(alpha=alpha)
    #
    #     result_dict, res_str = evaluator_test.evaluateRecommender(recommender)
    #     MAP_per_k_test.append(result_dict[10]["MAP"])
    #
    #     result_dict, res_str = evaluator_valid.evaluateRecommender(recommender)
    #     MAP_per_k_valid.append(result_dict[10]["MAP"])
    #
    # pyplot.plot(x_tick, MAP_per_k_test)
    # pyplot.ylabel('MAP_test')
    # pyplot.xlabel('alpha')
    # pyplot.show()
    #
    # pyplot.plot(x_tick, MAP_per_k_valid)
    # pyplot.ylabel('MAP_valid')
    # pyplot.xlabel('alpha')
    # pyplot.show()

    earlystopping_keywargs = {"validation_every_n": 5,
                              "stop_on_validation": True,
                              "evaluator_object": evaluator_test,
                              "lower_validations_allowed": 2,
                              "validation_metric": "MAP"
                              }

    recommender = SLIM_BPR_Cython(urm_train)
    recommender.fit(epochs=60, lambda_i=0.0297, lambda_j=0.0188, learning_rate=0.008083, topK=202, **earlystopping_keywargs)

    res = recommender.best_validation_metric
    n_epochs = recommender.epochs_best

    recommender = SLIM_BPR_Cython(urm_train)
    recommender.fit(epochs=n_epochs, lambda_i=0.0297, lambda_j=0.0188, learning_rate=0.008083, topK=202)








    # recommender = ItemKNNCFRecommender(urm_train)
    # recommender.fit(shrink=30, topK=20)



    normal_recommender = recommender

if test:

    if temperature == 'cold':
        result, str_result = evaluator_test.evaluateRecommender(cold_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))

    if temperature == 'zero':
        result, str_result = evaluator_test.evaluateRecommender(zero_recommender)
        print('The Map of test is : {}'.format(result[10]['MAP']))
        if valid:
            result, str_result = evaluator_valid.evaluateRecommender(zero_recommender)
            print('The Map of valid is : {}'.format(result[10]['MAP']))
    if temperature == 'warm':
        result, str_result = evaluator_test.evaluateRecommender(warm_recommender)
        print('The Map is : {}'.format(result[10]['MAP']))
        if valid:
            result, str_result = evaluator_valid.evaluateRecommender(warm_recommender)
            print('The Map of valid is : {}'.format(result[10]['MAP']))

    if temperature == 'normal':
        result, str_result = evaluator_test.evaluateRecommender(normal_recommender)
        print('The Map of test is : {}'.format(result[10]['MAP']))
        # if valid:
        #     result, str_result = evaluator_valid.evaluateRecommender(normal_recommender)
        #     print('The Map of valid is : {}'.format(result[10]['MAP']))
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
    submission.to_csv(data_folder / 'Data/Submissions/15-12_submission.csv', index=False)

