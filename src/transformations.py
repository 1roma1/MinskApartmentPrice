import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def get_transformer():
    cat = ['house_type', 'district', 'bathroom', 'balcony']
    num = ['year_built', 'floor', 'ceiling_height', 'total_area',
           'living_area', 'kitchen_area', 'number_of_rooms']

    transformer = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(), cat),
    ])

    return transformer


def transform_data(transformer, X_train, X_test, y_train, y_test):
    X_train_tr = transformer.fit_transform(X_train)
    X_test_tr = transformer.transform(X_test)
    y_train_tr = np.log1p(y_train)
    y_test_tr = np.log1p(y_test)
    return X_train_tr, X_test_tr, y_train_tr, y_test_tr


