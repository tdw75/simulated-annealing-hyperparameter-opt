import numpy as np


def get_lightgbm_parameters(objective='binary'):
    params_const = {
        'objective': [objective],
        'class_weight': [None]
    }
    params = {
        'learning_rate': [round(x, 4) for x in np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=15)],
        'num_leaves': [2, 20, 50, 100, 150],
        'max_depth': [0, 5, 6, 7, 8, 10, 15, 20],
        'colsample_bytree': [round(x, 1) for x in np.linspace(0.1, 1, num=10).round(1)],
        'subsample': [round(x, 1) for x in np.linspace(0.1, 1, num=10).round(1)],
        'reg_lambda': [0] + [round(x, 4) for x in np.logspace(-2, 1, base=10, num=4)],
        'reg_alpha': [0] + [round(x, 4) for x in np.logspace(-2, 1, base=10, num=4)],
        'min_child_samples': [10, 20, 50, 100, 250],
        'boosting_type': ['dart', 'goss', 'gbdt']
    }
    params.update(params_const)

    return params


def get_xgboost_parameters(objective='binary'):
    params_const = {
        'objective': [objective],
        'class_weight': [None]
    }
    params = {
        'learning_rate': [round(x, 4) for x in np.logspace(np.log(0.005), np.log(0.2), base=np.exp(1), num=15)],
        'n_estimators': [50, 100, 200, 250, 500],
        'max_depth': [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
        'colsample_bytree': [round(x, 1) for x in np.linspace(0.1, 1, num=10).round(1)],
        'subsample': [round(x, 1) for x in np.linspace(0.1, 1, num=10).round(1)],
        'reg_lambda': [0] + [round(x, 4) for x in np.logspace(-2, 1, base=10, num=4)],
        'min_child_weight': [10, 20, 50, 100, 250],
        'booster': ['dart', 'gblinear ', 'gbtree']
    }
    params.update(params_const)

    return params


def get_random_forest_parameters():
    pass
