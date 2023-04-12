import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
warnings.simplefilter('ignore', UserWarning)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np 
from PyImpetus import PPIMBC
import src.preprocessing.feature_extraction.text.filtering as filter 
import src.preprocessing.feature_extraction.text.wrapping as wrapping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
from numpy.typing import ArrayLike
from timeit import default_timer as timer
import logging
import json
import sys
from functools import partial
from datetime import timedelta
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


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
    accuracy_list=[]
    for result in results_list:
        for cls, metrics_dict in result['classification_report_train'].items():
            if cls == 'accuracy':
                accuracy_list.append(metrics_dict)
                continue
            for metric, value in metrics_dict.items():
                if cls not in metrics_list:
                    metrics_list[cls] = {}
                if metric not in metrics_list[cls]:
                    metrics_list[cls][metric] = []
                metrics_list[cls][metric].append(value)
    results_summarized['classification_report_train']['accuracy']={'mean': np.mean(accuracy_list), 'std': np.std(accuracy_list)/np.sqrt(len(metrics_list))}

    for cls, metrics in metrics_list.items():
        if cls not in results_summarized['classification_report_train']:
            results_summarized['classification_report_train'][cls] = {}
        for metric, values in metrics.items():
            results_summarized['classification_report_train'][cls][f'{metric}_mean'] = np.mean(values)
            results_summarized['classification_report_train'][cls][f'{metric}_std'] = np.std(values)/np.sqrt(len(values))

    accuracy_list=[]
    metrics_list = {}
    for result in results_list:
        for cls, metrics_dict in result['classification_report_test'].items():
            if cls == 'accuracy':
                accuracy_list.append(metrics_dict)
                continue
            for metric, value in metrics_dict.items():
                if cls not in metrics_list:
                    metrics_list[cls] = {}
                if metric not in metrics_list[cls]:
                    metrics_list[cls][metric] = []
                metrics_list[cls][metric].append(value)

    results_summarized['classification_report_test']['accuracy']={'mean': np.mean(accuracy_list), 'std': np.std(accuracy_list)/np.sqrt(len(metrics_list))}

    for cls, metrics in metrics_list.items():
        if cls not in results_summarized['classification_report_test']:
            results_summarized['classification_report_test'][cls] = {}
        for metric, values in metrics.items():
            results_summarized['classification_report_test'][cls][f'{metric}_mean'] = np.mean(values)
            results_summarized['classification_report_test'][cls][f'{metric}_std'] = np.std(values)/np.sqrt(len(values))

    timings_summarized = {}
    for key, value in timings.items():
        timings_summarized[key] = str(np.mean(value))
    
    results_summarized['timing'] = timings_summarized
    if ('n_words' in results_list[0]):
        results_summarized['n_words'] = []
        for i in results_list:
            results_summarized['n_words'].append(i['n_words'])

    if 'selected_vocabulary' in results_list[0]:
        results_summarized['selected_vocabulary'] = [result['selected_vocabulary'] for result in results_list]

    return results_summarized

def test_ppfs_extractor(model: sklearn.base.BaseEstimator, df: pd.DataFrame) -> dict:
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []
    X = df.drop(columns=['Class'])
    y = df['Class']

    k_fold = StratifiedKFold(n_splits=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, (train_idx, test_idx) in enumerate(k_fold.split(X, y)):
            print(f'PPFS: {idx+1} cv run out of 10.')
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            start = timer()
            ppfs_model = PPIMBC(DecisionTreeClassifier(random_state=27), p_val_thresh=0.05, num_simul=30, cv=5, n_jobs=-1, verbose=0)
            X_train_filtered = ppfs_model.fit_transform(X_train, y_train.values)
            end = timer()
            timing['extractor_fit'].append(timedelta(seconds=end-start))
            logging.info('Fit extractor.')

            start = timer()
            X_test_filtered = ppfs_model.transform(X_test)
            features_filtered = ppfs_model.MB
            end = timer()
            timing['filtered_features'].append(timedelta(seconds=end-start))

            start = timer()
            model.fit(X_train_filtered, y_train)
            end = timer()    
            timing['model_training_time'].append(timedelta(seconds=end-start))
            logging.info('Model training finished.')

            results, timing = record_results(model=model, 
                                        X_t=X_train_filtered,
                                        y_t=y_train,
                                        X_val=X_test_filtered,
                                        y_val=y_test,
                                        timing=timing)
            results['n_words'] = len(ppfs_model.MB)
            results['selected_vocabulary'] = features_filtered
            results_list.append(results)
    
    results = summarize_results(results_list, timing)
    return results

def shap_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, n_features: int) -> dict:
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

    X = df.drop(columns=['Class'])
    y = df['Class']

    group_k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in group_k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]


        extractor = wrapping.ShapFeatureExtractor(vocabulary=X.columns)

        start = timer()
        extractor.fit(X_train, y_train)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        X_train_filtered, filtered_features = extractor.filter_n_best(X_train, n_features)
        X_test_filtered, _ = extractor.filter_n_best(X_test, n_features)
        end = timer()
        timing['filtered_features'].append(timedelta(seconds=end-start))
    
        start = timer()
        model.fit(X_train_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_filtered,
                                    y_t=y_train,
                                    X_val=X_test_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_features
        results['selected_vocabulary'] = filtered_features.tolist()
        results_list.append(results)
    
    results = summarize_results(results_list, timing)

    return results

def test_extractor(model: sklearn.base.BaseEstimator, extractor: filter.BaseTextFeatureExtractor, df: pd.DataFrame, n_features: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    timing['extractor_fit'] = []
    timing['filtered_features'] = []
    timing['model_training_time'] = []
    results_list = []

    X = df.drop(columns=['Class'])
    y = df['Class']

    group_k_fold = StratifiedKFold(n_splits=5)

    for train_idx, test_idx in group_k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        start = timer()
        extractor.fit(X_train, y_train)
        end = timer()
        timing['extractor_fit'].append(timedelta(seconds=end-start))
        logging.info('Fit extractor.')
        
        start = timer()
        features = X.columns.values
        X_train_filtered, features_filtered = extractor.filter_n_best(X_train, n_features, features)
        X_test_filtered, _ = extractor.filter_n_best(X_test, n_features, features)
        end = timer()
        timing['filtered_features'].append(timedelta(seconds=end-start))


        start = timer()
        model.fit(X_train_filtered, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train_filtered,
                                    y_t=y_train,
                                    X_val=X_test_filtered,
                                    y_val=y_test,
                                    timing=timing)
        results['n_words'] = n_features
        results['selected_vocabulary'] = features_filtered.tolist()
        results_list.append(results)
    
    results = summarize_results(results_list, timing)

    return results


def get_baseline(df: pd.DataFrame, model: sklearn.base.BaseEstimator) -> dict:
    timing = {}
    timing['model_training_time'] = []
    results_list = []

    X = df.drop(columns=['Class'])
    y = df['Class']

    group_k_fold = StratifiedKFold(n_splits=10)

    for train_idx, test_idx in group_k_fold.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        start = timer()
        model.fit(X_train, y_train)
        end = timer()    
        timing['model_training_time'].append(timedelta(seconds=end-start))
        logging.info('Model training finished.')

        results, timing = record_results(model=model, 
                                    X_t=X_train,
                                    y_t=y_train,
                                    X_val=X_test,
                                    y_val=y_test,
                                    timing=timing)
        results_list.append(results)
    results = summarize_results(results_list, timing)                        
    return results

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


df = pd.read_csv('data/arcene/arcene.csv', sep=';', index_col=0)
# df = df.dropna()
df['Class'] = df['Class'] - 1

model = SVC()
result = test_ppfs_extractor(model, df)
with open(f'results/arcene/results_ppfs.json', 'w') as file:
                json.dump(result, file) 


n_words_options = [10, 100, 500, 1000, 3000, 5000, 8000]
filter_extractors = {}
filter_extractors['mutual_information'] = filter.MutualInformationFeatureExtractor()
filter_extractors['f_val'] = filter.FValFeatureExtractor()

method_list = {}
for name, extractor in filter_extractors.items():
    method_list[name] = partial(test_extractor, model, extractor, df)

method_list['shap'] = partial(shap_based_method, df, model)

for n_words in n_words_options:
    for name, method in method_list.items():
        print(f'Testing {name} at {n_words} features.')
        result = method(n_words)
        with open(f'results/arcene/results_{name}_{n_words}.json', 'w') as file:
            json.dump(result, file) 


result = get_baseline(df, model)
with open(f'results/arcene_baseline.json', 'w') as file:
    json.dump(result, file)