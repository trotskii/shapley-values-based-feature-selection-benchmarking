import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import  CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
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

    predictions_test = model.predict(X_val)

    report_train = classification_report(y_t, predictions_train, output_dict=True)
    report_test = classification_report(y_val, predictions_test, output_dict=True)

    # cm_train = confusion_matrix(y_t, predictions_train, normalize='true')
    # cm_test = confusion_matrix(y_val, predictions_test, normalize='true')

    results = {}
    results['training_data_samples'] = X_t.shape[0]
    results['test_data_samples'] = X_val.shape[0]
    results['classification_report_train'] = report_train
    results['classification_report_test'] = report_test
    # results['confustion_matrix_train'] = cm_train.tolist()
    # results['confusion_matrix_test'] = cm_test.tolist()
    results['model_type'] = type(model).__name__

    return results, timing

def summarize_results(results_list, timings):
    results_summarized = {}
    results_summarized['classification_report_train'] = {}
    results_summarized['classification_report_test'] = {}
    metrics_list = {}
    for result in results_list:
        for cls, metrics_dict in result['classification_report_train'].items():
            if cls == 'accuracy':
                continue
            for metric, value in metrics_dict.items():
                if cls not in metrics_list:
                    metrics_list[cls] = {}
                if metric not in metrics_list[cls]:
                    metrics_list[cls][metric] = []
                metrics_list[cls][metric].append(value)

    for cls, metrics in metrics_list.items():
        if cls not in results_summarized['classification_report_train']:
            results_summarized['classification_report_train'][cls] = {}
        for metric, values in metrics.items():
            results_summarized['classification_report_train'][cls][f'{metric}_mean'] = np.mean(values)
            # was np.square(len(values))
            results_summarized['classification_report_train'][cls][f'{metric}_std'] = np.std(values)/np.sqrt(len(values))


    metrics_list = {}
    for result in results_list:
        for cls, metrics_dict in result['classification_report_test'].items():
            if cls == 'accuracy':
                continue
            for metric, value in metrics_dict.items():
                if cls not in metrics_list:
                    metrics_list[cls] = {}
                if metric not in metrics_list[cls]:
                    metrics_list[cls][metric] = []
                metrics_list[cls][metric].append(value)

    for cls, metrics in metrics_list.items():
        if cls not in results_summarized['classification_report_test']:
            results_summarized['classification_report_test'][cls] = {}
        for metric, values in metrics.items():
            results_summarized['classification_report_test'][cls][f'{metric}_mean'] = np.mean(values)
            # was missing
            results_summarized['classification_report_test'][cls][f'{metric}_std'] = np.std(values)/np.sqrt(len(values))

    timings_summarized = {}
    for key, value in timings.items():
        timings_summarized[key] = str(np.mean(value))
    
    results_summarized['timing'] = timings_summarized

    if 'selected_vocabulary' in results_list[0]:
        results_summarized['selected_vocabulary'] = [result['selected_vocabulary'] for result in results_list]

    return results_summarized


def test_extractor(model: sklearn.base.BaseEstimator, extractor: filter.BaseTextFeatureExtractor, df: pd.DataFrame, n_words: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []

    X = df['Text']
    y = df['Label']
    k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        count_vectorizer = CountVectorizer(binary=True)
        count_vectorizer.fit(X_train)

        X_train_vectorized = count_vectorizer.transform(X_train)
        X_test_vectorized = count_vectorizer.transform(X_test)

        start = timer()
        extractor.fit(X_train_vectorized, y_train)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        vocabulary = count_vectorizer.get_feature_names_out()
        X_train_vectorized_filtered, vocabulary_filtered = extractor.remove_n_best(X_train_vectorized, n_words, vocabulary)
        X_test_vectorized_filtered, _ = extractor.remove_n_best(X_test_vectorized, n_words, vocabulary)
        end = timer()
        timing['filtered_features'].append(timedelta(seconds=end-start))


        tfidf = TfidfTransformer()
        X_train_vectorized_filtered = tfidf.fit_transform(X_train_vectorized_filtered, y_train)
        X_test_vectorized_filtered = tfidf.transform(X_test_vectorized_filtered)

        start = timer()
        model.fit(X_train_vectorized_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_vectorized_filtered,
                                    y_t=y_train,
                                    X_val=X_test_vectorized_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_words
        results['selected_vocabulary'] = vocabulary_filtered.tolist()
        results_list.append(results)

    results = summarize_results(results_list, timing)
    return results

def tfidf_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, n_words: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []

    X = df['Text']
    y = df['Label']
    k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        count_vectorizer = CountVectorizer(binary=True)
        count_vectorizer.fit(X_train)

        X_train_vectorized = count_vectorizer.transform(X_train)
        X_test_vectorized = count_vectorizer.transform(X_test)

        extractor = filter.CTFIDFFeatureExtractor()
        df_extractor_train = pd.DataFrame(X_train, columns=['Text'])
        df_extractor_train['Label'] = y_train
        start = timer()
        extractor.fit(df_extractor_train)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        vocabulary = count_vectorizer.get_feature_names_out()
        X_train_vectorized_filtered, vocabulary_filtered = extractor.remove_n_best(X_train_vectorized, n_words, vocabulary)
        X_test_vectorized_filtered, _ = extractor.remove_n_best(X_test_vectorized, n_words, vocabulary)
        end = timer()
        timing['filtered_features'].append(timedelta(seconds=end-start))

        tfidf = TfidfTransformer()
        X_train_vectorized_filtered = tfidf.fit_transform(X_train_vectorized_filtered, y_train)
        X_test_vectorized_filtered = tfidf.transform(X_test_vectorized_filtered)

        start = timer()
        model.fit(X_train_vectorized_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_vectorized_filtered,
                                    y_t=y_train,
                                    X_val=X_test_vectorized_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_words
        results['selected_vocabulary'] = vocabulary_filtered.tolist()
        results_list.append(results)
    
    results = summarize_results(results_list, timing)
    return results

def shap_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, n_words: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []

    X = df['Text']
    y = df['Label']
    k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        count_vectorizer = CountVectorizer(binary=True)
        count_vectorizer.fit(X_train)

        X_train_vectorized = count_vectorizer.transform(X_train)
        X_test_vectorized = count_vectorizer.transform(X_test)
        
        vocabulary = count_vectorizer.get_feature_names_out()
        extractor = wrapping.ShapFeatureExtractor(vocabulary=vocabulary)

        start = timer()
        extractor.fit(X_train_vectorized, y_train)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        X_train_vectorized_filtered, vocabulary_filtered = extractor.remove_n_best(X_train_vectorized, n_words)
        X_test_vectorized_filtered, _ = extractor.remove_n_best(X_test_vectorized, n_words)
        end = timer()
        timing['filtered_features'].append(timedelta(seconds=end-start))
        
        tfidf = TfidfTransformer()
        X_train_vectorized_filtered = tfidf.fit_transform(X_train_vectorized_filtered, y_train)
        X_test_vectorized_filtered = tfidf.transform(X_test_vectorized_filtered)

        start = timer()
        model.fit(X_train_vectorized_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_vectorized_filtered,
                                    y_t=y_train,
                                    X_val=X_test_vectorized_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_words
        results['selected_vocabulary'] = vocabulary_filtered.tolist()
        results_list.append(results)
    results = summarize_results(results_list, timing)

    return results

def lfs_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, n_words: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []

    X = df['Text']
    y = df['Label']
    k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        count_vectorizer = CountVectorizer(binary=True)
        count_vectorizer.fit(X_train)

        X_train_vectorized = count_vectorizer.transform(X_train)
        X_test_vectorized = count_vectorizer.transform(X_test)
        
        vocabulary = count_vectorizer.get_feature_names_out()
        ranker = filter.Chi2FeatureExtractor()
        extractor = wrapping.LinearForwardSearch(model, ranker, vocabulary)

        start = timer()
        extractor.fit(X_train_vectorized, y_train, k=50, n_words=n_words)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        X_train_vectorized_filtered = X_train_vectorized[:, extractor.selected_idx]
        X_test_vectorized_filtered = X_test_vectorized[:, extractor.selected_idx]
        timing['filtered_features'].append(timedelta(seconds=end-start))

        tfidf = TfidfTransformer()
        X_train_vectorized_filtered = tfidf.fit_transform(X_train_vectorized_filtered, y_train)
        X_test_vectorized_filtered = tfidf.transform(X_test_vectorized_filtered)

        start = timer()
        model.fit(X_train_vectorized_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_vectorized_filtered,
                                    y_t=y_train,
                                    X_val=X_test_vectorized_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_words
        results['selected_vocabulary'] = extractor.get_selected_words_lfs().tolist()
        results_list.append(results)
    
    results = summarize_results(results_list, timing)
    return results

def get_baseline(df: pd.DataFrame, model: sklearn.base.BaseEstimator) -> dict:
    timing = {}
    timing['model_training_time'] = []
    results_list = []

    X = df['Text']
    y = df['Label']
    k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        count_vectorizer = CountVectorizer(binary=True)
        count_vectorizer.fit(X_train)

        X_train_vectorized = count_vectorizer.transform(X_train)
        X_test_vectorized = count_vectorizer.transform(X_test)

        tfidf = TfidfTransformer()
        X_train_vectorized = tfidf.fit_transform(X_train_vectorized, y_train)
        X_test_vectorized = tfidf.transform(X_test_vectorized)

        start = timer()
        model.fit(X_train_vectorized, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_vectorized,
                                    y_t=y_train,
                                    X_val=X_test_vectorized,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = X_train_vectorized.shape[1]
        results_list.append(results)
    
    results = summarize_results(results_list, timing)

    return results


def main():
    df = pd.read_csv('data/brown_corpus/brown_corpus.csv', sep=';')
    df = df.fillna('')
    df = df.astype('str')

    df['Label'] = df['Label'].astype('category')
    df['Label'] = df['Label'].cat.codes
    df['Text'] = df['Text'].apply(tp.normalize_text)
    print('Finished preprocessing.')
    model = OneVsRestClassifier(SVC(class_weight='balanced', kernel='rbf', gamma=1/10))

    # n_words_options = [50, 100, 200, 500, 1000, 3000, 5000, 10000, 15000, 25000]
    n_words_options = [100, 500, 1000, 5000, 10000, 20000]
    filter_extractors = {}
    filter_extractors['term_strength'] = filter.TermStrengthFeatureExtractor()
    filter_extractors['mutual_information'] = filter.MutualInformationFeatureExtractor()
    filter_extractors['chi2'] = filter.Chi2FeatureExtractor()
    # filter_extractors['trl'] = filter.TRLFeatureExtractor()
    # filter_extractors['eccd'] = filter.ECCDFeatureExtractor()
    # filter_extractors['linear_measure_5'] = filter.LinearMeasureBasedFeatureExtractor(k=5)
    method_list = {}
    for name, extractor in filter_extractors.items():
        method_list[name] = partial(test_extractor, model, extractor, df)

    method_list['shap'] = partial(shap_based_method, df, model)
    method_list['tfidf'] = partial(tfidf_based_method, df, model)
    for n_words in n_words_options:
        for name, method in method_list.items():
            start = timer()
            print(f'Testing {name} at {n_words} words: ', end='', flush=True)
            result = method(n_words)
            end = timer()
            result = method(n_words)
            print(f'took: {str(timedelta(seconds=end-start))}')
            with open(f'results/brown_remove/results_{name}_{n_words}.json', 'w') as file:
                json.dump(result, file) 
    
    result = get_baseline(df, model)
    with open(f'results/brown_baseline.json', 'w') as file:
        json.dump(result, file) 

if __name__ == '__main__':
    main()