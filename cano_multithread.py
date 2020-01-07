


from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import numpy as np
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Hybrid.HybridNormRecommender import HybridNormRecommender
from DataManager.split_train_validation_leave_k_out import split_train_leave_k_out_user_wise
from DataManager.DataManager import DataManager
from Hybrid.HybridNormOrigRecommender import HybridNormOrigRecommender
from pathos.multiprocessing import ProcessingPool as Pool

data = DataManager()


tuning_params = dict()
tuning_params = {
  "alpha": (0, 1),
  "beta": (0, 1),
  "gamma": (0, 1),
  "phi": (0, 1),
  "psi": (0, 1),
  "li": (0, 1),
 }



# def search_param(alpha, beta, gamma, phi, psi, li):
#     recommender.fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)
#     result_test, str_result = evaluator_test.evaluateRecommender(recommender)
#     result_valid, str_result = evaluator_valid.evaluateRecommender(recommender)
#     print('Il Map del test è : {}'.format(result_test[10]['MAP']))
#     return result_valid[10]['MAP']

num_test = 5

n_recommender = []
n_urm_valid = []

for i in np.arange(num_test):

    urm_train, urm_valid = split_train_leave_k_out_user_wise(data.get_urm(), temperature='normal')
    urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid2')
    recommender = HybridNormOrigRecommender(urm_train)
    n_recommender.append(recommender)
    n_urm_valid.append(urm_valid)





def my_func(alpha, beta, gamma, phi, psi, li):
    def single_test(i):
        evaluator_valid = EvaluatorHoldout(n_urm_valid[i], cutoff_list=[10])
        n_recommender[i].fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)
        result, str_result = evaluator_valid.evaluateRecommender(n_recommender[i])
        return result[10]['MAP']

    pool = Pool(6)
    res = pool.map(single_test, range(num_test))
    print('Il max è : {}'.format(optimizer.max))
    res = np.array(res)
    print('Il Map è : {}'.format(res.mean()))
    return res.mean()


def search_param(alpha, beta, gamma, phi, psi, li):
    return my_func(alpha, beta, gamma, phi, psi, li)


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=50,
)

logger = JSONLogger(path="./Logs/tmp/" + 'HybridNormRecommender' + ".json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.probe(
    params={'alpha': 0.1376138293683269, 'beta': 0.9543691669578935, 'gamma': 0.013053892649820154, 'li': 0.03484338600571313, 'phi': 0.93535595412577, 'psi': 0.09992511680395422},
    lazy=True,
)

optimizer.maximize(
    init_points=15,
    n_iter=500,
)

print(optimizer.max)
