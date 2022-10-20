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
            self.shap_values = None
            self.feature_strength_metric = None
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
        self.feature_strength_metric = np.mean(np.abs(shap_values), axis=0) 
    
    def get_n_words_shap(self, n_words):
        """
        Get n_words most important words from vocabulary per class according to shap values. Will return duplicates per class label for consistency with other methods.
        """


        return self.vocabulary[self.feature_strength_metric.argsort()[-n_words:]]
    
    def filter_n_best(self, X, n_best):
        """
        Leave n_best terms.
        Arguments:
            X - dataset to filter out terms from 
            n_best - number of terms to leave
        Returns:
            X_filtered - filtered dataset
            vocabulary_filtered - vocabulary of the new filtered dataset (will have length of n_best)
        """
        selected_index = self.feature_strength_metric.argsort()[-n_best:]
        X_filtered = X[:, selected_index]
        vocabulary_filtered = self.vocabulary[selected_index]

        return X_filtered, vocabulary_filtered