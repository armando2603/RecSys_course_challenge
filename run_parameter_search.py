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

Data = DataManager()


urm_temp, urm_test = split_train_leave_k_out_user_wise(Data.get_urm(),
                                                                        use_validation_set=False, leave_random_out=True)

urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_temp,
                                                                        use_validation_set=False, leave_random_out=True)


tuning_params = dict()
tuning_params = {
    "NN": (20, 600),
    "BA": (1, 20),
    "EP": (20, 100)
 }


def search_param(NN, BA, EP):
    recommender = SLIM_BPR_Cython(urm_train)
    recommender.fit(batch_size=int(BA), epochs=int(EP), topK=int(NN))
    res_valid = evaluate(urm_valid, recommender)

    return res_valid["MAP"]


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=2,
)

optimizer.maximize(
    init_points=20,
    n_iter=100,
)
