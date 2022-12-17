import pandas as pd 
import numpy as np 
import scipy
import sklearn 
from timeit import default_timer as timer
from datetime import timedelta
import logging
import json
import sys
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from numpy.typing import ArrayLike

import src.preprocessing.text_preprocessing as tp 
import src.preprocessing.feature_extraction.text.filtering as filter 
import src.preprocessing.feature_extraction.text.wrapping as wrapping

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)



def record_results(model: sklearn.base.BaseEstimator, X_t: ArrayLike, y_t: ArrayLike, X_val: ArrayLike, y_val: ArrayLike, timing: dict) -> dict:
    """
    Evaluates the passed model both on test and train set and returns a dict with evaluation results.
    Arguments:
        model - trained sklearn model to evaluate
        X_t - training features
        y_t - training labels
        X_val - testing features
        y_val - testing labels
        timing - dict with timings to add inference timings to
    Retruns:
        results - dict with all evaluation results
    """

    predictions_train = model.predict(X_t)

    start = timer()
    predictions_test = model.predict(X_val)
    end = timer()
    timing['model_inference_time'] = str(timedelta(seconds=end-start))
    logging.info('Model inference finished.')

    report_train = classification_report(y_t, predictions_train, output_dict=True)
    report_test = classification_report(y_val, predictions_test, output_dict=True)

    cm_train = confusion_matrix(y_t, predictions_train, normalize='true')
    cm_test = confusion_matrix(y_val, predictions_test, normalize='true')

    results = {}
    results['timing'] = timing
    results['training_data_samples'] = X_t.shape[0]
    results['test_data_samples'] = X_val.shape[0]
    results['classification_report_train'] = report_train
    results['classification_report_test'] = report_test
    results['confustion_matrix_train'] = cm_train.tolist()
    results['confusion_matrix_test'] = cm_test.tolist()
    results['model_type'] = type(model).__name__
    results['model_params'] = model.get_params() # cannot serialize for onevsrest

    return results


def test_extractor(model: sklearn.base.BaseEstimator, extractor: filter.BaseTextFeatureExtractor, df: pd.DataFrame, split: float, n_features: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)



    start = timer()
    extractor.fit(X_train, y_train)
    end = timer()
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    features = X.columns.values
    X_train_filtered, features_filtered = extractor.filter_n_best(X_train, n_features, features)
    X_test_filtered, _ = extractor.filter_n_best(X_test, n_features, features)
    end = timer()
    timing['filtered_features'] = str(timedelta(seconds=end-start))


    start = timer()
    model.fit(X_train_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_filtered,
                                y_t=y_train,
                                X_val=X_test_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_features
    results['selected_vocabulary'] = features_filtered.tolist()

    return results

def shap_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float, n_features: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)

    extractor = wrapping.ShapFeatureExtractor(vocabulary=X.columns)

    start = timer()
    extractor.fit(X_train, y_train)
    end = timer()
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    X_train_filtered, filtered_features = extractor.filter_n_best(X_train, n_features)
    X_test_filtered, _ = extractor.filter_n_best(X_test, n_features)
    end = timer()
    timing['filtered_features'] = str(timedelta(seconds=end-start))
   
    start = timer()
    model.fit(X_train_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_filtered,
                                y_t=y_train,
                                X_val=X_test_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_features
    results['selected_vocabulary'] = filtered_features.tolist()

    return results

def lfs_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float, n_words: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)


    ranker = filter.MutualInformationFeatureExtractor()
    extractor = wrapping.LinearForwardSearch(model, ranker, X.columns.values)

    start = timer()
    extractor.fit(X_train, y_train, k=10, n_words=n_words)
    end = timer()
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    X_train_filtered = X_train.iloc[:, extractor.selected_idx]
    X_test_filtered = X_test.iloc[:, extractor.selected_idx]
    timing['filtered_features'] = str(timedelta(seconds=end-start))

    start = timer()
    model.fit(X_train_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_filtered,
                                y_t=y_train,
                                X_val=X_test_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words
    results['selected_vocabulary'] = extractor.get_selected_words_lfs().tolist()

    return results

def get_baseline(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float) -> dict:
    timing = {}
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)

    start = timer()
    model.fit(X_train, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train,
                                y_t=y_train,
                                X_val=X_test,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = X_train.shape[1]
    return results

def main():
    df = pd.read_csv('data/mitdb_data/mitdb_ecg.csv', sep=';', index_col=0)


    model = SVC(gamma='auto', class_weight='balanced')

    n_words_options = [2,3,5,10,15,21]
    filter_extractors = {}
    filter_extractors['mutual_information'] = filter.MutualInformationFeatureExtractor()
    filter_extractors['f_val'] = filter.FValFeatureExtractor()

    method_list = {}
    for name, extractor in filter_extractors.items():
        method_list[name] = partial(test_extractor, model, extractor, df, 0.2)

    method_list['shap'] = partial(shap_based_method, df, model, 0.2)
    method_list['lfs'] = partial(lfs_based_method, df, model, 0.2)

    # for n_words in n_words_options:
    #     for name, method in method_list.items():
    #         print(f'Testing {name} at {n_words} features.')
    #         result = method(n_words)
    #         with open(f'results/ecg_mit/results_{name}_{n_words}.json', 'w') as file:
    #             json.dump(result, file) 


    result = get_baseline(df, model, 0.2)
    with open(f'results/mit_bih_baseline.json', 'w') as file:
        json.dump(result, file) 
if __name__ == '__main__':
    main()