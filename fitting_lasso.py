import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet

from utils import fit_result, get_parameters

np.random.seed(42)

dataset = pd.read_csv('./datasets/3_512_x_main.csv')
target = pd.read_csv('./datasets/3_512_y_main.csv')
x_values = dataset.values
y_values = target.values.ravel()


# import fitting_settings.json to retrive optimization hyperparameters for
# raidus = 3, fingerprint bits = 512 dataset
dic = get_parameters(path = './settings/fitting_settings.json', print_dict = False)

##### Lasso
fit_result(x_values, y_values, Lasso(alpha=0.01, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))

#selection cyclic
fit_result(x_values, y_values, Lasso(selection = 'cyclic', alpha=0.001, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'cyclic', alpha=0.01, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'cyclic', alpha=0.1, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'cyclic', alpha=1., max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'cyclic', alpha=10, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))

fit_result(x_values, y_values, Lasso(selection = 'random', alpha=0.001, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'random', alpha=0.01, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'random', alpha=0.1, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'random', alpha=1., max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))
fit_result(x_values, y_values, Lasso(selection = 'random', alpha=10, max_iter=100000), random_seeds_start = dic.get("random_seeds_start"), random_seeds_stop = dic.get("random_seeds_stop"), random_seeds_step = dic.get("random_seeds_step"))