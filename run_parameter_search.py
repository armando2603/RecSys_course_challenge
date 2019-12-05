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

Data = DataManager()


urm_temp, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(),
                                                                        use_validation_set=False, leave_random_out=True)

urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_temp,
                                                                        use_validation_set=False, leave_random_out=True)

recommender = UserKNNCFRecommender(urm_train)

tuning_params = dict()
tuning_params = {
    "TK": (120, 250),
    "SH": (1, 20),
 }




def search_param(TK, SH):
    recommender.fit(topK=TK, shrink=SH)
    res_valid = evaluate(urm_valid, recommender)
    #evaluate(urm_test, recommender)
    return res_valid["MAP"]


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=5,
)

#load_logs(optimizer, logs=["./logs.json"])


logger = JSONLogger(path="./logsUserCF.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

# optimizer.probe(
#     params={"NN": 21,
#     "BA": 1,
#     "EP": 399,
#     "LE": 0.0007,
#     "L1": 0.0444,
#     "L2": 0.02658285},
#     lazy=True,
# )


optimizer.maximize(
    init_points=15,
    n_iter=40,
)

print(optimizer.max)

