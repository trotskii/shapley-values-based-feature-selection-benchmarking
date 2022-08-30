import pandas as pd 
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix
from scipy.special import comb
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest

class TermStrengthFeatureExtractor:
    """
    Calculates term strength according to http://mlwiki.org/index.php/Term_Strength
    """
    def __init__(self):
        self.term_strength = {}
        self.classes = []

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

        for cls in classes:
            idx = np.where(y == cls)[0] # might need [0] to unbox the tuple
            X_cls = X[idx]
            n_docs, _ = X_cls.shape

            n_t = X_cls.sum(axis=0) # number of documents with term in question (each column for each term)
            nominator = comb(n_t, 2) # N of pairs where t in both docs
            denominator = comb(n_docs, 2) - comb(n_docs-n_t, 2) # number of pairs where t in at least one doc
            s_t = np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator!=0)
            self.term_strength[cls] = s_t
    
    def transform(self, X):
        """
        Transform word presense into term strength. Returns s_t for each class for each word.
        Arguments:
            X - binary output from CountVectorizer
        Returns:
            s_t - dict with a structure {class_label : transformed features as csr_matrix}
        """
        X = X.asfptype()
        s_t = {}
        for cls in self.classes:
            s_t[cls] = csr_matrix(X.minimum(np.tile(self.term_strength[cls], (X.shape[0], 1)))) # sets term (0 or 1) to term strength, term strength is <= 1, so element wise min works
        
        return s_t 

    def prune(self, n_std):
        """
        Prune weak terms. Prune term t if s(t) <= std(s)*n_std*E[s(t)]
        Arguments:
            n_std - pruning param
        Returns:
            keep_idx - dict {class_label:pruned_idx}, where prune_idx is an array with remaining indexes 
        """
        keep_idx = {}
        for cls in self.classes:
            E_t = np.mean(self.term_strength[cls])
            std = np.std(self.term_strength[cls])
            keep_idx[cls] = np.where(self.term_strength[cls] > n_std*std*E_t)[1] 

        return keep_idx

    def get_n_strongest_terms_words(self, n_words, vocabulary):
        """
        Get n_words most important words from vocabulary per class according to term strength values.
        """
        words_per_class = {label: [vocabulary[index] for index in self.term_strength[label][0].argsort()[-n_words:]] for label in self.classes}

        return words_per_class

            
class MutualInformationFeatureExtractor:
    """
    Mutual Information based feature extractor. Implemented based on arXiv:1509.07577
    """
    def __init__(self):
        self.mutual_information = {}
        self.classes = []

    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X -  counts of word (output from CountVectorizer)
            y - array-like with class labels for X
        """
        if isinstance(y, np.ndarray):
            classes = np.unique(y) 
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            classes = y.unique()
        else:
            raise ValueError(f'Unexpected type for y: {type(y)}. y must be array like')
        
        self.classes = classes

        mi = mutual_info_classif(X, y, discrete_features=True)

        self.mutual_information = mi 
    
    def transform(self, X):
        return csr_matrix(X.minimum(np.tile(self.mutual_information, (X.shape[0], 1))))

    def get_n_words_mi(self, n_words, vocabulary):
        """
        Get n_words most important words from vocabulary per class according to mutual information values. Will return duplicates per class label for consistency with other methods.
        """
        words_per_class = {label: [vocabulary[index] for index in self.mutual_information.argsort()[-n_words:]] for label in self.classes}

        return words_per_class


class Chi2FeatureExtractor:
    """
    Chi - square based feature exctractor. 
    """
    def __init__(self):
        self.chi2_values = {}
        self.classes = []
    
    def fit(self, X, y):
        """
        Fit feature exctractor
        Arguments:
            X - counts of words (output from CountVectorizer)
            y - array-like with class labels for X
        """
        if isinstance(y, np.ndarray):
            classes = np.unique(y) 
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            classes = y.unique()
        else:
            raise ValueError(f'Unexpected type for y: {type(y)}. y must be array like')

        self.classes = classes

        chi2_values = chi2(X, y)[0] # pick only chi2 values and discard p-values
        self.chi2_values = chi2_values
    
    def get_n_words_chi2(self, n_words, vocabulary):
        """
        Get n_words most important words from vocabulary per class according to chi2 values. Will return duplicates per class label for consistency with other methods.
        """
        words_per_class = {label: [vocabulary[index] for index in self.chi2_values.argsort()[-n_words:]] for label in self.classes}

        return words_per_class
