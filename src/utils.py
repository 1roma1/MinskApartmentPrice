import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_report(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def train_test_score(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_train)
    train_scores = regression_report(y_train, y_pred)
    y_pred = model.predict(X_test)
    test_scores = regression_report(y_test, y_pred)
    df = pd.DataFrame([train_scores, test_scores], index=['Train', 'Test'])
    return df


def get_predict_pipeline(transformer, model):
    predict_pipeline = Pipeline([
        ("preparation", transformer),
        ("est", model),
    ])
    return predict_pipeline