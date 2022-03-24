import joblib
import argparse
import pandas as pd

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_data
from src.transformations import transform_data, get_transformer
from src.optimization import xgb_optimize, lgb_optimize, cb_optimize
from src.utils import get_predict_pipeline, train_test_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, default="",
        choices=['xgb', 'lgb', 'cb'], help="type of training model")
    ap.add_argument("-d", "--data", type=str, default="",
        help="path to dataset file")
    ap.add_argument("-s", "--save", type=str, default="",
        help="path to model file")
    ap.add_argument('-o', '--optimize', action='store_true',
        help="whether or not optimize model hyperparameters")
    args = vars(ap.parse_args())    

    if args['data'] == "":
        path = 'data/minsk_flats.csv'
    else:
        path = args['path']

    df = pd.read_csv("data/analysed_dataset.csv")

    # X = df.drop("target_price", axis=1)
    # y = df["target_price"]

    X, y = preprocess_data(path)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1)

    transformer = get_transformer()
    X_train_tr, X_test_tr, y_train_tr, y_test_tr = \
        transform_data(transformer, X_train, X_test, y_train, y_test)

    if args['model'] == 'xgb':
        model = XGBRegressor(n_estimators=20)
    elif args['model'] == 'lgb':
        model = LGBMRegressor(n_estimators=30)
    elif args['model'] == 'cb':
        model = CatBoostRegressor(verbose=0)
    
    model.fit(X_train_tr, y_train_tr)
    scores = train_test_score(model, X_train_tr, X_test_tr, y_train_tr, y_test_tr)

    print(model.__class__.__name__)
    print(scores)

    if args['optimize']:
        if args['model'] == 'xgb':
            model = xgb_optimize(model, X_train_tr, y_train_tr)
        elif args['model'] == 'lgb':
            model = lgb_optimize(model, X_train_tr, y_train_tr)
        elif args['model'] == 'cb':
            model = cb_optimize(model, X_train_tr, y_train_tr)

        model.fit(X_train_tr, y_train_tr)
        scores = train_test_score(model, X_train_tr, X_test_tr, y_train_tr, y_test_tr)

        print("\nAfter hyperparameters tuning")
        print(scores)

    if args['save'] != "":
        predict_pipeline = get_predict_pipeline(transformer, model)
        joblib.dump(predict_pipeline, args['save'])
