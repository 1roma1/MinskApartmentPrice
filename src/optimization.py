import numpy as np

from sklearn.model_selection import GridSearchCV


def xgb_optimize(model, X, y, params=None):
    if params is None:
        params = {
            'n_estimators': np.arange(20, 40, 5),
            'max_depth': np.arange(3, 9, 2),
            'learning_rate': np.arange(0.05, 0.2, 0.05),
        }
    xgb_grid_search = GridSearchCV(model, params, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    xgb_grid_search.fit(X, y)
    return xgb_grid_search
    

def lgb_optimize(model, X, y, params=None):
    if params is None:
        params = {
            'n_estimators': np.arange(20, 40, 5),
            'num_leaves': np.arange(10, 40, 5), 
            'max_depth': [5, 7, 9]
        }
    lgb_grid_search = GridSearchCV(model, params, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    lgb_grid_search.fit(X, y)
    return lgb_grid_search


def cb_optimize(model, X, y, params=None):
    if params is None:
        params = {
            'depth': [6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.02, 0.03, 0.04],
            'iterations': [70, 80, 90, 100]
        }
    cb_grid_search = GridSearchCV(model, params, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
    cb_grid_search.fit(X, y)
    return cb_grid_search