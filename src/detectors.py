import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance


def train_xgb(X_train, y_train, **kwargs):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train, **kwargs):
    model = MLPClassifier(max_iter=200, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_if(X_train, contamination=0.1):
    model = IsolationForest(contamination=contamination)
    model.fit(X_train)
    return model


def train_maha(X_train, y_train):
    cov = EmpiricalCovariance().fit(X_train[y_train == 0])
    return cov


def get_detector_predictions(model, X, detector_type, frac=None):
    if detector_type == 'IF':
        scores = -model.decision_function(X)
        y_pred = model.predict(X)
        y_pred = np.where(y_pred == -1, 1, 0)
    elif detector_type == 'Maha':
        dists = model.mahalanobis(X)
        scores = dists
        if frac is not None:
            thresh = np.quantile(dists, 1 - frac)
            y_pred = (dists > thresh).astype(int)
        else:
            y_pred = (dists > np.median(dists)).astype(int)
    else:
        scores = model.predict_proba(X)[:, 1]
        y_pred = (scores >= 0.5).astype(int)
    return y_pred, scores
