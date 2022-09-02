import pandas as pd 
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap


class ShapFeatureExtractor:
    """
    Calculates shap feature importances based on xgboost classifier.
    """

    def __init__(self, vocabulary):
            self.shap_values = {}
            self.classes = []
            self.vocabulary = vocabulary
    
    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X - word counts (output from CountVectorizer)
            y - array-like with class labels for X
        """
        if isinstance(y, np.ndarray):
            classes = np.unique(y) 
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            classes = y.unique()
        else:
            raise ValueError(f'Unexpected type for y: {type(y)}. y must be array like')
        
        self.classes = classes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_test)

        dmatrix = xgb.DMatrix(X, label=y)
        param = {'max_depth':5, 'eta':0.1, 'lambda': 0.01, 'objective':'binary:logistic' }

        model = xgb.train(param, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train'), (test_dmatrix, 'test')], num_boost_round=1000, early_stopping_rounds=100, verbose_eval=False)

        explainer = shap.TreeExplainer(model, feature_names=self.vocabulary)
        shap_values = explainer.shap_values(test_dmatrix)

        self.shap_values = shap_values
    
    def get_n_words_shap(self, n_words):
        """
        Get n_words most important words from vocabulary per class according to shap values. Will return duplicates per class label for consistency with other methods.
        """

        shap_feature_importance = np.mean(np.abs(self.shap_values), axis=0) 

        words_per_class = {label: [self.vocabulary[index] for index in shap_feature_importance.argsort()[-n_words:]] for label in self.classes}

        return words_per_class
