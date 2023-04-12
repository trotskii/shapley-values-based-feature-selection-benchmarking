import pandas as pd 
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        test_dmatrix = xgb.DMatrix(X_test, label=y_test)

        if len(classes) > 2:
            param = {'max_depth':25, 'eta':0.05, 'lambda': 0.001, 'objective':'multi:softprob', 'num_class': len(classes) }
        else:
            param = {'max_depth':25, 'eta':0.05, 'lambda': 0.001, 'objective':'binary:logistic' }
            
        model = xgb.train(param, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train'), (test_dmatrix, 'test')], num_boost_round=1000, early_stopping_rounds=100, verbose_eval=False)

        explainer = shap.TreeExplainer(model, feature_names=self.vocabulary)
        shap_values = explainer.shap_values(test_dmatrix, check_additivity=False)

        self.shap_values = shap_values


        if len(classes) > 2:
            # for multiclass classification, shap values are unique per each category, so we take the max 
            # across the classes (just like with filtering methods, where applicable)
            shap_vals_avg_per_class_list = []
            for cls in shap_values:
                shap_vals_avg_per_class_list.append(np.mean(np.abs(cls), axis=0) )
            shap_vals_avg_per_class = np.vstack(shap_vals_avg_per_class_list)
            self.feature_strength_metric = np.maximum.reduce(shap_vals_avg_per_class)
        else: 
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
        if isinstance(X, pd.DataFrame):
            X_filtered = X.iloc[:, selected_index]
        else:
            X_filtered = X[:, selected_index]
        vocabulary_filtered = self.vocabulary[selected_index]

        return X_filtered, vocabulary_filtered

    def remove_n_best(self, X, n_words):
        """
        Remove n_best features.
            Arguments:
            X - dataset to filter out terms from
            n_words - [int] number of terms to remove
            vocabulary - [list] list of words present in the dataset
        Returns:
            X_filtered - filtered dataset
            vocabulary_filtered - dropped words
        """
        # selected_index = self.feature_strength_metric.argsort()[:-n_words]
        selected_index = self.feature_strength_metric.argsort()[n_words:]
        dropped_index = self.feature_strength_metric.argsort()[-n_words:]

        if isinstance(X, pd.DataFrame):
            X_filtered = X.iloc[:, selected_index]
        else:
            X_filtered = X[:, selected_index]
        vocabulary_filtered = self.vocabulary[dropped_index]

        return X_filtered, vocabulary_filtered


class LinearForwardSearch():
    
    estimator = None
    ranker = None
    vocabulary = None
    selected_idx = None
    epsillon = 1E-4 

    def __init__(self, estimator, ranker, vocabulary):
        """
        Linear Forward Search feature extractor. Ref: https://researchcommons.waikato.ac.nz/handle/10289/2205
        Arguments:
            estimator - estimator to use for feature subset evaluation
            ranker - filtering feature extractor to use for initial feature ranking
            vocabulary - list of words present in the dataset
        """
        self.estimator = estimator
        self.ranker = ranker
        self.vocabulary = vocabulary

    def forward_search_step(self, X, y, k, R, selected_index):
        estimator = sklearn.base.clone(self.estimator)
        ranked_features_idx = R[-(k+len(selected_index)):]
        features_to_check = [idx for idx in ranked_features_idx if idx not in selected_index]

        best_score = 0
        best_idx = -1

        for feature in features_to_check:
            curr_feature_list = np.append(selected_index, feature)
            if isinstance(X, pd.DataFrame):
                X_ = X.iloc[:, curr_feature_list]
            else:
                X_ = X[:,curr_feature_list]
            scores = cross_val_score(estimator, X_, y, cv=5, scoring='roc_auc', n_jobs=-1)
            score = np.mean(scores)
            if score > best_score:
                best_score = score 
                best_idx = feature
        
        return np.append(selected_index, best_idx).astype(int), best_score



    def fit(self, X, y, k, n_words = None):
        self.ranker.fit(X, y)
        R = self.ranker.feature_strength_metric.argsort()

        if n_words is None:
            selected_idx = []
            best_score = 0
            for i in range(X.shape[1]):
                selected_idx, score = self.forward_search_step(X, y, k, R, selected_idx)
                if (score - best_score) < self.epsillon:
                    break
                best_score = score
        else:
            selected_idx = []
            for i in range(n_words):
                selected_idx, score = self.forward_search_step(X, y, k, R, selected_idx)
        
        self.selected_idx = R[selected_idx]
        return self.selected_idx
    
    def get_selected_words_lfs(self):
        return self.vocabulary[self.selected_idx]
        

