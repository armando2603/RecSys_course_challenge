"""
Created on 04/12/19

@author: Giuseppe Serna
"""
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout
from MatrixFactorization.IALSRecommender import IALSRecommender
from ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from skopt.space import Real, Integer, Categorical
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
import numpy as np
import scipy.sparse as sps
from FeatureWeighting.User_CFW_D_Similarity_Linalg import User_CFW_D_Similarity_Linalg
from Hybrid.HybridGen2Recommender import HybridGen2Recommender
from Hybrid.HybridNormRecommender import HybridNormRecommender
from Hybrid.HybridNorm1Recommender import HybridNorm1Recommender

Data = DataManager()


urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), threshold=10, temperature='normal')
urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, threshold=10, temperature='valid')
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

recommender = HybridNorm1Recommender

# recommender_3 = UserKNNCFRecommender(urm_train)
# recommender_3.fit(shrink=2, topK=600, normalize=True)
# w_sparse = recommender_3.W_sparse

parameterSearch = SearchBayesianSkopt(recommender,
                                 evaluator_validation=evaluator_valid,
                                 evaluator_test=evaluator_test)

# earlystopping_keywargs = {"validation_every_n": 5,
#                               "stop_on_validation": True,
#                               "evaluator_object": evaluator_valid,
#                               "lower_validations_allowed": 2,
#                               "validation_metric": "MAP"
#                           }

hyperparameters_range_dictionary = {}
hyperparameters_range_dictionary["beta"] = Real(0, 1)
# hyperparameters_range_dictionary["gamma"] = Real(0, 1)
# hyperparameters_range_dictionary["phi"] = Real(0, 1)

# hyperparameters_range_dictionary["topK"] = Integer(5, 100)
# hyperparameters_range_dictionary["shrink"] = Integer(0, 500)
# hyperparameters_range_dictionary["feature_weighting"] = Categorical(["BM25", "none", "TF-IDF"])
# hyperparameters_range_dictionary["similarity"] = Categorical(["tversky"])
# hyperparameters_range_dictionary["normalize"] = Categorical([True, False])
# hyperparameters_range_dictionary["tversky_alpha"] = Real(0, 1)
# hyperparameters_range_dictionary["tversky_beta"] = Real(0, 1)

# hyperparameters_range_dictionary = {}
# hyperparameters_range_dictionary["topK"] = Integer(5, 2000)
# hyperparameters_range_dictionary["add_zeros_quota"] = Real(low = 0, high = 1, prior = 'uniform')
# hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])


# ucm_w = sps.load_npz('Data/ucm_weighted.npz')
# ucm_age, ucm_region, ucm_all = Data.get_ucm()

recommender_input_args = SearchInputRecommenderArgs(
    CONSTRUCTOR_POSITIONAL_ARGS=[urm_train],
    CONSTRUCTOR_KEYWORD_ARGS={},
    FIT_POSITIONAL_ARGS=[],
    FIT_KEYWORD_ARGS={}
)

output_folder_path = "result_experiments/"

import os

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

n_cases = 300
metric_to_optimize = "MAP"
parameterSearch.search(recommender_input_args,
                       parameter_search_space = hyperparameters_range_dictionary,
                       n_cases = n_cases,
                       n_random_starts = 10,
                       output_folder_path = output_folder_path,
                       output_file_name_root = recommender.RECOMMENDER_NAME,
                       metric_to_optimize = metric_to_optimize
                      )


# def search_param(**tuning_params):
#     # recommender.fit(epochs=400, topK=int(tuning_params['NN']), lambda_i=tuning_params['L1'],
#     #                 lambda_j=tuning_params['L2'], learning_rate=tuning_params['LE'], **earlystopping_keywargs)
#     recommender.fit(epochs=30, alpha=tuning_params['A'], num_factors=int(tuning_params['NF']), reg=tuning_params['REG'], **earlystopping_keywargs)
#     #res_test, str = evaluator_test.evaluateRecommender(recommender)
#     res_test, str = evaluator_valid.evaluateRecommender(recommender)
#     return res_test[10]["MAP"]

# optimizer = BayesianOptimization(
#     f=search_param,
#     pbounds=tuning_params,
#     verbose=3,
#     random_state=5,
# )

#load_logs(optimizer, logs=["./logs.json"])

#
# logger = JSONLogger(path="./Logs/tmp/" + recommender.RECOMMENDER_NAME + ".json")
# optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
#
# optimizer.probe(
#     params={
#     'A': 5.0,
#     'NF': 99.9773715259928,
#     'REG': 0.0189810660740337},
#     lazy=True,
# )


#
# optimizer.maximize(
#     init_points=15,
#     n_iter=50,
# )
#
# print(optimizer.max)

