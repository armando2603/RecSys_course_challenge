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
from GraphBased.RP3betaRecommender import RP3betaRecommender
data = DataManager()


tuning_params = dict()
tuning_params = {
    "alpha": (0, 1),
    "beta": (0, 1),
    'topK': (0, 50),
    # "gamma": (0, 1),
    # "phi": (0, 1),
    # "psi": (0, 1),
    # "li": (0, 1),
    # "mi": (0, 1),
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
n_urm_test = []
for i in np.arange(num_test):

    urm_train, urm_test = split_train_leave_k_out_user_wise(data.get_urm(), temperature='normal')
    urm_train, urm_valid = split_train_leave_k_out_user_wise(urm_train, temperature='valid')
    recommender = RP3betaRecommender(urm_train)
    n_recommender.append(recommender)
    n_urm_valid.append(urm_valid)
    n_urm_test.append(urm_test)

vec = {'n': 0, 'max_test':0, 'max_valid':0, 'n_test':0 , 'n_valid':0}

def my_func(alpha, beta, topK):
    def single_valid(i):
        evaluator_valid = EvaluatorHoldout(n_urm_valid[i], cutoff_list=[10])
        #n_recommender[i].fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)
        n_recommender[i].fit(alpha=alpha, beta=beta, topK=int(topK))
        result, str_result = evaluator_valid.evaluateRecommender(n_recommender[i])
        return result[10]['MAP']

    def single_test(i):
        evaluator_test = EvaluatorHoldout(n_urm_test[i], cutoff_list=[10])
        #n_recommender[i].fit(alpha=alpha, beta=beta, gamma=gamma, phi=phi, psi=psi, li=li)
        n_recommender[i].fit(alpha=alpha, beta=beta, topK=int(topK))
        result, str_result = evaluator_test.evaluateRecommender(n_recommender[i])
        return result[10]['MAP']

    pool = Pool(5)
    res = pool.map(single_valid, range(num_test))

    res = np.array(res)

    print('Il max valid è il n: {}  con : {}'.format(vec['n_valid'], optimizer.max))
    print('Il max test è il n : {} con test : {}'.format(vec['n_test'], vec['max_test']))
    res = np.array(res)
    print('Il Map corrente è il n : {} con : {}'.format(vec['n'], res.mean()))

    if res.mean() > vec['max_valid']:
        vec['n_valid'] = vec['n']
        vec['max_valid'] = res.mean()
        print('new max valid found')
        pool = Pool(6)
        res_test = pool.map(single_test, range(num_test))
        res_test = np.array(res_test)
        res_test = np.array(res_test)
        if res_test.mean() > vec['max_test']:
            print('un nuovo max è stato trovato')
            vec['max_test'] = res_test.mean()
            vec['n_test'] = vec['n']
    vec['n'] += 1
    return res.mean()


def search_param(alpha, beta, topK):
    return my_func(alpha, beta, topK)


optimizer = BayesianOptimization(
    f=search_param,
    pbounds=tuning_params,
    verbose=3,
    random_state=2010,
)

logger = JSONLogger(path="./Logs/tmp/" + 'RP3beta_multi' + ".json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.probe({
    'alpha': 0.03374950051351756,
    'beta': 0.24087176329409027,
    'topK': 16},
    lazy=True,
)

optimizer.maximize(
    init_points=40,
    n_iter=400,
)

print(optimizer.max)
