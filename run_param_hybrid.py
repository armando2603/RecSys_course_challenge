"""
Created on 04/12/19

@author: Giuseppe Serna
"""
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Evaluator.evaluation import evaluate
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch
from Notebooks_utils.data_splitter import train_test_holdout
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Hybrid.HybridRecommender import HybridRecommender
from Hybrid.HybridPredRecommender import HybridPredRecommender
from pathlib import Path


Data = DataManager()


urm_train, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(), use_validation_set=False, leave_random_out=True)

# urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, use_validation_set=False, leave_random_out=True)

# urm_train, urm_test = train_test_holdout(Data.get_urm(), train_perc=0.8)
# urm_train, urm_valid = train_test_holdout(urm_train, train_perc=0.8)
evaluator_valid = EvaluatorHoldout(urm_valid, cutoff_list=[10])
# evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10])

recommender = HybridPredRecommender(urm_train)

tuning_params = dict()
tuning_params = {
    "A": (0.3, 0.5),
    "B": (0.1, 0.4),
    "G": (0.01, 0.1)
    # "NN": (40, 300),
 }


def search_param(A, B, G):
    recommender.fit(alpha=A, beta=B, gamma=G)
    res_test = evaluate(urm_test, recommender)
    return res_test["MAP"]


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=5,
)

#load_logs(optimizer, logs=["./logs.json"])



logger = JSONLogger(path='Logs/tmp/'+recommender.RECOMMENDER_NAME+'.json')
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)



optimizer.maximize(
    init_points=15,
    n_iter=20,
)

print(optimizer.max)

