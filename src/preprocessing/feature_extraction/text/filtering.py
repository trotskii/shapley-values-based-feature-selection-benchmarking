import pandas as pd 
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix
from scipy.special import comb
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.metrics import mutual_info_score
from src.preprocessing.ctfidf import CTFIDFVectorizer
from sklearn.preprocessing import minmax_scale
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import sys

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
        if isinstance(X, pd.DataFrame):
            X_filtered = X.iloc[:, selected_index]
        else:
            X_filtered = X[:, selected_index]
        vocabulary_filtered = vocabulary[selected_index]

        return X_filtered, vocabulary_filtered
    
    def remove_n_best(self, X, n_words, vocabulary):
        """
        Remove n_best features.
            Arguments:
            X - dataset to filter out terms from
            n_words - [int] number of terms to remove
            vocabulary - [list] list of words present in the dataset
        Returns:
            X_filtered - filtered dataset
            vocabulary_filtered - removed words
        """
        # selected_index = self.feature_strength_metric.argsort()[:-n_words]
        selected_index = self.feature_strength_metric.argsort()[n_words:]

        dropped_index = self.feature_strength_metric.argsort()[-n_words:]

        if isinstance(X, pd.DataFrame):
            X_filtered = X.iloc[:, selected_index]
        else:
            X_filtered = X[:, selected_index]
        vocabulary_filtered = vocabulary[dropped_index]

        return X_filtered, vocabulary_filtered

class CTFIDFFeatureExtractor(BaseTextFeatureExtractor):
    """
    Calculate feature strength according to ctfidf.
    """
    def __init__(self):
        self.feature_strength_metric = None 
    
    def fit(self, df):
        text_per_class = df.groupby(['Label'], as_index=False).agg({'Text': ' '.join})
        count_vectorizer = CountVectorizer().fit(text_per_class['Text'])
        count = count_vectorizer.transform(text_per_class['Text'])

        ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=df.shape[0]).toarray()
        labels = text_per_class['Label'].unique()

        feature_strength = []
        for label in labels:
            feature_strength.append(ctfidf[label])
        self.feature_strength_metric = np.squeeze(np.maximum.reduce(feature_strength))

    def transform(self, X):
        """
        Transforms input to corresponding feature importance values
        """
        X_t = X.copy()
        feature_st_matrix = np.tile(self.feature_strength_metric, (X.shape[0], 1))
        X_t = np.copyto(X_t, feature_st_matrix, where=X_t != 0)
        
        return X_t 



class TermStrengthFeatureExtractor():
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

class mRMR(BaseTextFeatureExtractor):
    """
    Mutual Information based feature extractor that also consideres relevance and redundancy. Implemented based on 10.1109/TPAMI.2005.159
    """

    def __init__(self):
        self.selected_indices = None
    
    @staticmethod
    def calc_opt_metric(X, y, idx_sel, idx):
        """
        Calculate optimization target metric
        Arguments:
            X - full dataset
            y - target variable
            idx_sel - indicies for the already selected features
            idx - index of the variable to check
        Returns:
            optimization target metric
        """
        X_j = X.iloc[:, idx]
        m = idx_sel.shape[0]
        # avoid 0 division
        if m < 2: 
            m = 2 
        I_xj_c = mutual_info_score(X_j, y)
        I_xj_xi_sum = 0
        for i in idx_sel:
            I_xj_xi_sum += mutual_info_score(X_j, X.iloc[:, i])
        I_xj_xi_sum = I_xj_xi_sum/(m-1)

        return I_xj_c - I_xj_xi_sum



    def fit(self, X, y, m):
        """
        Fit feature extractor.
        Arguments:
            X - full dataset
            y - target variable
            m - number of feature to select
        """
        all_idx = np.arange(X.shape[1])
        idx_sel = np.array([]).astype(int)
        for i in range(m):
            idx_unchecked = all_idx[np.isin(all_idx, idx_sel,invert=True)]
            opt_metric = [mRMR.calc_opt_metric(X, y, idx_sel, idx) for idx in idx_unchecked]
            selected = np.argmax(opt_metric)
            idx_sel = np.append(idx_sel, selected)

            sys.stdout.write("Variables checked %d out of %d   \r" % (i+1, m) )
            sys.stdout.flush()
        
        self.selected_indices = idx_sel.astype(int)
    
    def filter_n_best(self, X, n, vocabulary):
        X_filtered = X.iloc[:, self.selected_indices]
        selected_vocabulary = vocabulary[self.selected_indices]
        return X_filtered, selected_vocabulary

    def remove_n_best(self, X, n, vocabulary):
        X_filtered = np.delete(X, self.selected_indices, axis=1)
        selected_vocabulary = np.delete(vocabulary, self.selected_indices)
        return X_filtered, selected_vocabulary



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
        
        mi = mutual_info_classif(X, y, discrete_features='auto')
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


class TRLFeatureExtractor(BaseTextFeatureExtractor):
    """
    Term ReLatedness based text feature extractor 
    https://www.researchgate.net/publication/282006918_A_Supervised_Term_Selection_Technique_for_Effective_Text_Categorization
    """
    def __init__(self):
        self.feature_strength_metric = None 
    
    @staticmethod
    def _prob_class(y):
        """
        Calculate frequentist probabilities of a document belonging to a class
        Arguments:
            y - array-like with class labels for the documents
        Returns:
            P_C - dict[class_label: prob]
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        P_C = dict(zip(unique_classes, counts/y.size))
        
        return P_C
    
    @staticmethod
    def _prob_term(x):
        """
        Calculate frequentist probabilities of a document containing each term
        Arguments:
            x - counts of words (output from CountVectorizer)
        Returns:
            P_t - numpy array with probabilities
        """
        x = x.copy() # avoid overwriting counts
        x = x > 0 # we need binary mask
        counts = x.sum(axis=0)
        P_t = np.squeeze(np.asarray(counts/x.shape[0]))

        return P_t

    @staticmethod
    def _prob_term_class(x, y):
        """
        Calculate probabilities that a document belonging to class c contains term t.
        """    
        classes = np.unique(y)
        P_t_C = {}
        for cls in classes:
            idx = np.where(y == cls)[0]
            x_cls = x[idx]
            P_t_C[cls] = TRLFeatureExtractor._prob_term(x_cls)
        
        return P_t_C

    @staticmethod
    def _prob_term_not_class(x, y):
        """
        Calculate probabilities that a document belonging to class c contains term t.
        """    
        classes = np.unique(y)
        P_t_C = {}
        for cls in classes:
            idx = np.where(y != cls)[0]
            x_cls = x[idx]
            P_t_C[cls] = TRLFeatureExtractor._prob_term(x_cls)
        
        return P_t_C


    @staticmethod
    def _calc_class_entropy(P_C):
        """
        Calculate E(C) from the paper
        """
        E_C = {}
        for key, value in P_C.items():
            E_C[key] = -value*np.log(value)
        return E_C 

    @staticmethod
    def _calc_term_factor(P_t, P_C, P_t_C, P_t_not_C, E_C):
        """
        Calculate TF(t,C) from the paper.
        """
        tf = {}
        for cls, cls_prob in P_C.items():
            tf[cls] = minmax_scale(np.divide(np.minimum(P_t, cls_prob) - P_t_C[cls], np.maximum(P_t, cls_prob) - P_t_C[cls]) * np.divide(P_t - P_t_not_C[cls], P_t) * E_C[cls])
        return tf

    @staticmethod
    def _calc_term_category_ratio(P_t_C, P_C, E_C):
        """
        Calculate TCR(t,C) from the paper
        """
        tcr = {}
        for cls, cls_prob in P_C.items():
            tcr[cls] = minmax_scale(np.divide(1+P_t_C[cls], 1+cls_prob) * E_C[cls])
        return tcr

    @staticmethod
    def _calc_term_relative_frequency(P_t_C, P_t, E_C):
        """
        Calculate TRF from the paper
        """
        trf = {}
        for cls, cls_prob in P_t_C.items():
            trf[cls] = minmax_scale(np.divide(1 + cls_prob, 1 + P_t) * E_C[cls])
        return trf


    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X - counts of words (output from CountVectorizer)
            y - array-like with class labels for X
        """
        P_t_C = TRLFeatureExtractor._prob_term_class(X, y)
        P_t_not_C = TRLFeatureExtractor._prob_term_not_class(X, y)
        P_C = TRLFeatureExtractor._prob_class(y)
        P_t = TRLFeatureExtractor._prob_term(X)
        E_C = TRLFeatureExtractor._calc_class_entropy(P_C)
        T_F = TRLFeatureExtractor._calc_term_factor(P_t, P_C, P_t_C, P_t_not_C, E_C)
        TCR = TRLFeatureExtractor._calc_term_category_ratio(P_t_C, P_C, E_C)
        TRF = TRLFeatureExtractor._calc_term_relative_frequency(P_t_C, P_t, E_C)

        classes = np.unique(y)
        trl = [] # no need to track what class has what trl value
        for cls in classes:
            trl_cls = -np.ones(X.shape[1]) # the values should be in the range 0-1, so -1 will indicate if we miss anything
            # implementantion of conditional filling in TRL (equation 1 in the paper)
            # conditions are implemented in the same order as defined in the equation
            idx_cond_1 = np.where(P_t_C[cls] == 0)[0]
            trl_cls[idx_cond_1] = 1 

            idx_cond_2 = np.where((P_t_C[cls] == P_t) & (P_t == P_C[cls]))[0]
            trl_cls[idx_cond_2] = 0

            idx_cond_3 = np.where((P_t_C[cls] == P_t) & (P_t != P_C[cls]))[0]
            trl_cls[idx_cond_3] = (1 - TCR[cls])[idx_cond_3]

            idx_cond_4 = np.where((P_t_C[cls] == P_C[cls]) & (P_t != P_C[cls]))[0]
            trl_cls[idx_cond_4] = (1 - TRF[cls])[idx_cond_4]

            idx_cond_5 = np.where(trl_cls == -1)[0]
            trl_cls[idx_cond_5] = (1 - T_F[cls])[idx_cond_5]

            trl.append(trl_cls)
        
        true_trl = np.minimum.reduce(trl)
        self.feature_strength_metric = 1-true_trl # invert trl value, so that the higher the value, the better the term is
        # helps with consistency with other methods
        # trl is always between 0 and 1 with 0 being perfect feature and 1 being useless, so 1 - true_trl works

    def transform(self, X):
        """
        Transform word presense into term strength. Returns s_t for each class for each word.
        Arguments:
            X - binary output from CountVectorizer
        Returns:
            s_t - dict with a structure {class_label : transformed features as csr_matrix}
        """
        X = X.asfptype()
        return csr_matrix(X.minimum(np.tile(self.feature_strength_metric, (X.shape[0], 1)))) # sets term (0 or 1) to term strength, term strength is <= 1, so element wise min works

class ECCDFeatureExtractor(BaseTextFeatureExtractor):
    """
    ECCD text feature extractor: https://dl.acm.org/doi/10.1145/1982185.1982389
    """
    def __init__(self):
        self.feature_strength_metric = None

    @staticmethod
    def _calc_term_frequency(X, y):
        """
        Calculate term frequencies for each class.
        Arguments:
            X - word counts from CountVectorizer
            y - class labels
        Returns:
            tf - np.array with vstacked frequencies
        """
        classes = np.unique(y)
        X = X.copy() # avoid overwriting
        X = X > 0
        tf = []
        for cls in classes:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            tf.append(np.squeeze(np.asarray(np.divide(X_cls.sum(axis=0), X.sum(axis=0)))))

        return np.vstack(tf)

    @staticmethod
    def _calc_term_entropy(X, y):
        """
        Calculate Shannon entropy for each term.
        Arguments:
            X - word counts from CountVectorizer
            y - class labels
        Returns:
            E - numpy array with entropy values
        """
        classes = np.unique(y)
        tf = ECCDFeatureExtractor._calc_term_frequency(X, y)
        E = entropy(tf, axis=0, base=2)
        return E
    
    @staticmethod
    def _calc_P_t_C(X, y):
        """
        Calculate probability of observing a term in a document belonging to a category
        Arguments:
            X - word counts from CountVectorizer
            y - class labels
        Returns:
            P_t_C - dict[class label: probability of observing each term (np.array)]
        """
        classes = np.unique(y)
        P_t_C = {}
        X = X.copy() # avoid overwriting
        X = X > 0
        for cls in classes:
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            counts = np.asarray(X_cls.sum(axis=0))
            P_t_C[cls] = counts/X_cls.shape[0]
        return P_t_C

    @staticmethod
    def _calc_P_t_not_C(X, y):
        """
        Calculate probability of observing a term in a document not belonging to a category
        Arguments:
            X - word counts from CountVectorizer
            y - class labels
        Returns:
            P_t_C - dict[class label: probability of observing each term (np.array)]
        """
        classes = np.unique(y)
        P_t_C = {}
        X = X.copy() # avoid overwriting
        X = X > 0
        for cls in classes:
            idx = np.where(y != cls)[0]
            X_cls = X[idx]
            counts = np.asarray(X_cls.sum(axis=0))
            P_t_C[cls] = counts/X_cls.shape[0]
        return P_t_C

    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X - counts of words (output from CountVectorizer)
            y - array-like with class labels for X
        """
        E = ECCDFeatureExtractor._calc_term_entropy(X, y)
        P_t_C = ECCDFeatureExtractor._calc_P_t_C(X, y)
        P_t_not_C = ECCDFeatureExtractor._calc_P_t_not_C(X, y)
        E_max = np.max(E)

        classes = np.unique(y)
        ECCD = []
        for cls in classes:
            ECCD.append( np.multiply(P_t_C[cls] - P_t_not_C[cls], np.divide(E_max - E,E_max)))
        
        ECCD = np.maximum.reduce(ECCD)
        self.feature_strength_metric = np.squeeze(ECCD)
        
    def transform(self, X):
        X_t = X.copy()
        feature_st_matrix = np.tile(self.feature_strength_metric, (X.shape[0], 1))
        X_t = np.copyto(X_t, feature_st_matrix, where=X_t != 0)
        return X_t 


class LinearMeasureBasedFeatureExtractor(BaseTextFeatureExtractor):
    """
    Linear measure based feature extractor. Ref: https://ieeexplore.ieee.org/abstract/document/1490529
    """
    k = 1

    def __init__(self, k=1):
        self.k = k

    @staticmethod
    def calc_a(X, y):
        """
        Calculate number of documents of the category c in which word w appears
        """
        X_binary = X.copy()
        X_binary = X_binary > 0
        classes = np.unique(y)

        a = {}

        for cls in classes:
            idx = np.where(y == cls)
            X_cls = X_binary[idx]
            a[cls] = np.asarray(X_cls.sum(axis=0))
        
        return a

    @staticmethod
    def calc_b(X, y):
        """
        Calculate number of documents in which word w appears, but do not belong to c
        """
        X_binary = X.copy()
        X_binary = X_binary > 0
        classes = np.unique(y)

        b = {}

        for cls in classes:
            idx = np.where(y != cls)
            X_cls = X_binary[idx]
            b[cls] = np.asarray(X_cls.sum(axis=0))
        
        return b

    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X - counts of words (output from CountVectorizer)
            y - array-like with class labels for X
        """
        a = LinearMeasureBasedFeatureExtractor.calc_a(X, y)
        b = LinearMeasureBasedFeatureExtractor.calc_b(X, y)
        classes = np.unique(y)

        lm = []

        for cls in classes:
            lm.append(self.k * a[cls] - b[cls])
        
        lm = np.maximum.reduce(lm)
        self.feature_strength_metric = np.squeeze(lm) 


class FValFeatureExtractor(BaseTextFeatureExtractor):
    """
    ANOVA F-value based feature extractor for classification tasks
    """
    def __init__(self) -> None:
        self.feature_strength_metric = None 
    
    def fit(self, X, y):
        """
        Fit feature extractor
        Arguments:
            X - numerical features
            y - array-like with class labels for X
        """
        f_values = f_classif(X, y)[0] # discard p_values
        self.feature_strength_metric = f_values
    
    def transform(self, X):
        X_t = X.copy()
        feature_st_matrix = np.tile(self.feature_strength_metric, (X.shape[0], 1))
        X_t = np.copyto(X_t, feature_st_matrix, where=X_t != 0)
        return X_t   
    