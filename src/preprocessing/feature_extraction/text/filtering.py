import pandas as pd 
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix
from scipy.special import comb
from sklearn.feature_selection import mutual_info_classif, chi2
from src.preprocessing.ctfidf import CTFIDFVectorizer

class BaseTextFeatureExtractor:
    """
    Base class for text feature selection methods. Implements functionality for feature subset filtering and defines the basic API.
    """
    feature_strength_metric = None

    def fit(self, X, y):
        raise NotImplementedError("fit function must be implemented for feature extractor")
    
    def transform(self, X):
        raise NotImplementedError("transform function must be implemented for feature extractor")

    def get_n_strongest_features(self, n_words, vocabulary):
        """
        Get n_words strongest features according to the feature extractor.
        Arguments:
            n_words - [int] number of features to pick
            vocabulary - vocabulary of the dataset, used for fitting the feature exctractor
        Returns:
            strongest_features - [list] list of best terms
        """
        selected_index = self.feature_strength_metric.argsort()[-n_words:]
        strongest_features = vocabulary[selected_index]
        return strongest_features
    
    def filter_n_best(self, X, n_best, vocabulary):
        """
        Leave n_best terms.
        Arguments:
            X - dataset to filter out terms from
            n_best - [int] number of terms to keep
            vocabulary - [list] list of words present in the dataset
        Returns:
            X_filtered - filtered dataset
            vocabulary_filtered - vocabulary of the new filtered dataset (will have length of n_best)
        """
        selected_index = self.feature_strength_metric.argsort()[-n_best:]
        X_filtered = X[:, selected_index]
        vocabulary_filtered = vocabulary[selected_index]

        return X_filtered, vocabulary_filtered


class TermStrengthFeatureExtractor(BaseTextFeatureExtractor):
    """
    Calculates term strength according to http://mlwiki.org/index.php/Term_Strength
    The resulting term strength values are the max between different classes. This should make this metric extensible for multiclass classification. 
    """
    def __init__(self):
        self.feature_strength_metric = None

    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X -  presense of word (binary output from CountVectorizer)
            y - array-like with class labels for X
        """
        if isinstance(y, np.ndarray):
            classes = np.unique(y) 
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            classes = y.unique()
        else:
            raise ValueError(f'Unexpected type for y: {type(y)}. y must be array like')
        
        self.classes = classes
        term_strength = []
        for cls in classes:
            idx = np.where(y == cls)[0] # might need [0] to unbox the tuple
            X_cls = X[idx]
            n_docs, _ = X_cls.shape

            n_t = X_cls.sum(axis=0) # number of documents with term in question (each column for each term)
            nominator = comb(n_t, 2) # N of pairs where t in both docs
            denominator = comb(n_docs, 2) - comb(n_docs-n_t, 2) # number of pairs where t in at least one doc
            s_t = np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator!=0)
            term_strength.append(s_t)
        
        self.feature_strength_metric = np.squeeze(np.maximum.reduce(term_strength))
        

    
    def transform(self, X):
        """
        Transform word presense into term strength. Returns s_t for each class for each word.
        Arguments:
            X - binary output from CountVectorizer
        Returns:
            s_t - dict with a structure {class_label : transformed features as csr_matrix}
        """
        X = X.asfptype()
        X_s_t = csr_matrix(X.minimum(np.tile(self.feature_strength_metric, (X.shape[0], 1)))) # sets term (0 or 1) to term strength, term strength is <= 1, so element wise min works
        
        return X_s_t

            
class MutualInformationFeatureExtractor(BaseTextFeatureExtractor):
    """
    Mutual Information based feature extractor. Implemented based on arXiv:1509.07577
    """
    def __init__(self):
        self.feature_strength_metric = None

    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X -  counts of word (output from CountVectorizer)
            y - array-like with class labels for X
        """
        
        mi = mutual_info_classif(X, y, discrete_features=True)
        self.feature_strength_metric = mi 
    
    def transform(self, X):
        return csr_matrix(X.minimum(np.tile(self.feature_strength_metric, (X.shape[0], 1))))


class Chi2FeatureExtractor(BaseTextFeatureExtractor):
    """
    Chi - square based feature exctractor. 
    """
    def __init__(self):
        self.feature_strength_metric = None
    
    def fit(self, X, y):
        """
        Fit feature exctractor
        Arguments:
            X - counts of words (output from CountVectorizer)
            y - array-like with class labels for X
        """
        chi2_values = chi2(X, y)[0] # pick only chi2 values and discard p-values
        self.feature_strength_metric = chi2_values
    
    def transform(self, X):
        X_t = X.copy()
        feature_st_matrix = np.tile(self.feature_strength_metric, (X.shape[0], 1))
        X_t = np.copyto(X_t, feature_st_matrix, where=X_t != 0)
        return X_t    

