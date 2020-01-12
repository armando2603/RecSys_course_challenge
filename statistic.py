import numpy as np
from DataManager.DataManager import DataManager
import scipy.sparse as sps

data = DataManager()

_, ucm_all, _ = data.get_ucm()

features_per_user = np.ediff1d(ucm_all.indptr)

ucm_all = sps.csc_matrix(ucm_all)
users_per_feature = np.ediff1d(ucm_all.indptr)

features_per_users = np.sort(features_per_user)
# users_per_feature = np.sort(users_per_feature)

print(features_per_users.shape)
print(users_per_feature.shape)

import matplotlib.pyplot as pyplot


pyplot.plot(users_per_feature, 'ro')
pyplot.ylabel('Num features ')
pyplot.xlabel('Users Index')
pyplot.show()