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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
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
    results['model_params'] = model.get_params()

    return results

def test_extractor(model: sklearn.base.BaseEstimator, extractor: filter.BaseTextFeatureExtractor, df: pd.DataFrame, split: float, n_words: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)

    count_vectorizer = CountVectorizer(binary=True)
    count_vectorizer.fit(X_train)

    X_train_vectorized = count_vectorizer.transform(X_train)
    X_test_vectorized = count_vectorizer.transform(X_test)

    start = timer()
    extractor.fit(X_train_vectorized, y_train)
    end = timer()
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    vocabulary = count_vectorizer.get_feature_names_out()
    X_train_vectorized_filtered, vocabulary_filtered = extractor.filter_n_best(X_train_vectorized, n_words, vocabulary)
    X_test_vectorized_filtered, _ = extractor.filter_n_best(X_test_vectorized, n_words, vocabulary)
    end = timer()
    timing['filtered_features'] = str(timedelta(seconds=end-start))


    start = timer()
    model.fit(X_train_vectorized_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_filtered,
                                y_t=y_train,
                                X_val=X_test_vectorized_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words
    results['selected_vocabulary'] = vocabulary_filtered.tolist()

    return results
  


def tfidf_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float, n_words: int) -> dict:
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)

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
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    vocabulary = count_vectorizer.get_feature_names_out()
    X_train_vectorized_filtered, vocabulary_filtered = extractor.filter_n_best(X_train_vectorized, n_words, vocabulary)
    X_test_vectorized_filtered, _ = extractor.filter_n_best(X_test_vectorized, n_words, vocabulary)
    end = timer()
    timing['filtered_features'] = str(timedelta(seconds=end-start))


    start = timer()
    model.fit(X_train_vectorized_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_filtered,
                                y_t=y_train,
                                X_val=X_test_vectorized_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words
    results['selected_vocabulary'] = vocabulary_filtered.tolist()

    return results


def shap_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float, n_words: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)

    count_vectorizer = CountVectorizer(binary=True)
    count_vectorizer.fit(X_train)

    X_train_vectorized = count_vectorizer.transform(X_train)
    X_test_vectorized = count_vectorizer.transform(X_test)
    
    vocabulary = count_vectorizer.get_feature_names_out()
    extractor = wrapping.ShapFeatureExtractor(vocabulary=vocabulary)

    start = timer()
    extractor.fit(X_train_vectorized, y_train)
    end = timer()
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    X_train_vectorized_filtered, vocabulary_filtered = extractor.filter_n_best(X_train_vectorized, n_words)
    X_test_vectorized_filtered, _ = extractor.filter_n_best(X_test_vectorized, n_words)
    end = timer()
    timing['filtered_features'] = str(timedelta(seconds=end-start))
   
    start = timer()
    model.fit(X_train_vectorized_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_filtered,
                                y_t=y_train,
                                X_val=X_test_vectorized_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words
    results['selected_vocabulary'] = vocabulary_filtered.tolist()

    return results

def lfs_based_method(df: pd.DataFrame, model: sklearn.base.BaseEstimator, split: float, n_words: int) -> dict:
    """
    Train model with mutual information based features selection.
    """
    """
    Train passed model on a features selected by passed extractor.
    """
    timing = {}
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)

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
    timing['extractor_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit extractor.')
    
    start = timer()
    X_train_vectorized_filtered = X_train_vectorized[:, extractor.selected_idx]
    X_test_vectorized_filtered = X_test_vectorized[:, extractor.selected_idx]
    timing['filtered_features'] = str(timedelta(seconds=end-start))

    start = timer()
    model.fit(X_train_vectorized_filtered, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_filtered,
                                y_t=y_train,
                                X_val=X_test_vectorized_filtered,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words
    results['selected_vocabulary'] = extractor.get_selected_words_lfs().tolist()

    return results

def main():
    df = pd.read_csv('data/enron/enron_spam_data.csv', sep=',')
    df = df.fillna('')
    df = df.astype('str')
    
    df['Text'] = df.apply(lambda x: x['Subject'] + ', ' + x['Message'], axis=1)
    df['Label'] = np.where(df['Spam/Ham'].values == 'ham', 0, 1)
    df['Text'] = df['Text'].apply(tp.normalize_text)

    model = SVC()

    n_words_options = [10, 50, 100, 200, 500, 1000, 3000, 5000, 10000, 15000, 25000]
    filter_extractors = {}
    filter_extractors['term_strength'] = filter.TermStrengthFeatureExtractor()
    filter_extractors['mutual_information'] = filter.MutualInformationFeatureExtractor()
    filter_extractors['chi2'] = filter.Chi2FeatureExtractor()
    filter_extractors['trl'] = filter.TRLFeatureExtractor()
    filter_extractors['eccd'] = filter.ECCDFeatureExtractor()
    filter_extractors['linear_measure_5'] = filter.LinearMeasureBasedFeatureExtractor(k=5)

    method_list = {}
    for name, extractor in filter_extractors.items():
        method_list[name] = partial(test_extractor, model, extractor, df, 0.7)

    method_list['shap'] = partial(shap_based_method, df, model, 0.7)
    method_list['tfidf'] = partial(tfidf_based_method, df, model, 0.7)
    # method_list['lfs'] = partial(lfs_based_method, df, model, 0.7)

    for n_words in n_words_options:
        for name, method in method_list.items():
            print(f'Testing {name} at {n_words} words.')
            result = method(n_words)
            with open(f'results/enron/results_{name}_{n_words}.json', 'w') as file:
                json.dump(result, file) 




if __name__ == '__main__':
    main()