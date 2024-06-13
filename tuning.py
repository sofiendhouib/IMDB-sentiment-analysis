from itertools import product
from sklearn.linear_model import SGDClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
import numpy as np



def model_on_param_grid(model_class, param_grid):
    configs = product(*param_grid.values())# TODO use the scklearn function 
    return [model_class(**dict(zip(param_grid.keys(), values)))
                for values in configs]


def model_on_param_grid2(model_class, param_grid):
    deployed_param_grid = list(ParameterGrid(param_grid))
    return [model_class(**param_dict)
                for param_dict in deployed_param_grid]

def choose_best_model(cv_res, thd= 0.8, prefer_low_gap= False):
    """ Choose a model from the results of a cross validation
    """
    if prefer_low_gap:
        best_models = cv_res[cv_res["mean_test_score"] >= thd]
        assert len(best_models)>0, "Threshold is too high !"
        gap = best_models["mean_train_score"] - best_models["mean_test_score"]
        return best_models.index[gap.abs().argmin()]
    else:
        return cv_res.index[cv_res["mean_test_score"].argmax()]
    

# The following list is to specify models and parameter grids to tune on
models_param_grid_zip = [

    (SGDClassifier, {"alpha": np.logspace(-5, -2, 4),
                "class_weight": [None, {0: 1, 1: 10}],
                "penalty": ['l1', 'l2'],
                "loss": ['hinge', 'log_loss']}),
    (RandomForestClassifier, {"n_estimators": [50, 100]}),
    (MultinomialNB, {"alpha": np.logspace(-4, 0, 5)}),
    
]

def model_set_params(model, cv_res, ind):
    """ use the string provided by grid search 
    to set the pipeline parameters
    """
    params_dict = cv_res["params"][ind]
    if isinstance(params_dict, str): params_dict = eval(params_dict)
    model.set_params(**params_dict)
    return model
    