"""
Created on 04/12/19

@author: Giuseppe Serna
"""
from DataManager.DataManager import DataManager
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
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
recommender = SLIM_BPR_Cython(urm_train)

tuning_params = dict()
tuning_params = {
    "NN": (20, 600),
    "BA": (1, 10),
    "EP": (20, 200),
    "LE": (4, 7),
    "L1": (0.001, 0.1),
    "L2": (0.0001, 0.01)
 }


def search_param(NN, BA, EP, LE, L1, L2):
    recommender.fit(batch_size=int(BA), epochs=int(EP), topK=int(NN), learning_rate=float('1e-'+str(int(LE))),
                    lambda_i=L1, lambda_j=L2)
    res_valid = evaluate(urm_valid, recommender)
    return res_valid["MAP"]


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=5,
)

load_logs(optimizer, logs=["./logs.json"])


logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)


optimizer.maximize(
    init_points=10,
    n_iter=30,
)

print(optimizer.max)

evaluate(urm_test, recommender)
