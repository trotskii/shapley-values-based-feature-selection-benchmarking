import pandas as pd 
import numpy as np 
import scipy
import sklearn 
from timeit import default_timer as timer
from datetime import timedelta
import logging
import json
import sys
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from numpy.typing import ArrayLike

import src.preprocessing.text_preprocessing as tp 
import src.preprocessing.feature_extraction.text.filtering as filter 
import src.preprocessing.feature_extraction.text.wrapping as wrapping

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

def test_extractor(model: sklearn.base.BaseEstimator, extractor: filter.BaseTextFeatureExtractor, df: pd.DataFrame, n_words: int, split: float) -> dict:
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
    tfidf_transformer = TfidfTransformer()

    start = timer()
    X_train_vectorized_tfidf = tfidf_transformer.fit_transform(X_train_vectorized_filtered)
    X_test_vectorized_tfidf = tfidf_transformer.transform(X_test_vectorized_filtered)
    end = timer()
    timing['vectorized_time'] = str(timedelta(seconds=end-start))
    logging.info('Vectorized dataset.')

    start = timer()
    model.fit(X_train_vectorized_tfidf, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_tfidf,
                                y_t=y_train,
                                X_val=X_test_vectorized_tfidf,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words

    return results
  


def tfidf_based_method(df: pd.DataFrame, n_words: int, model: sklearn.base.BaseEstimator, split: float) -> dict:
    """
    Train passed model according to tfidf feature selection.
    Arguments: 
        df - dataframe with 'Lable' and 'Text' for training and validation
        n_words - number of important words per class to keep
        model - sklearn model
        split - train/test split (0,1)
    Retruns:
        results - dictionary with performance metrics
    """
    timing = {}

    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)

    df_train = pd.DataFrame(X_train)
    df_train['Label'] = y_train 

    df_test = pd.DataFrame(X_test)
    df_test['Label'] = y_test

    start = timer()
    words_per_class = tp.get_n_most_important_words_cftfidf(df_train, n_words=n_words)
    end = timer()
    timing['n_important_words_train_time'] = str(timedelta(seconds=end-start))
    logging.info('Got important words per class.')
    

    start = timer()
    filtered_train_df = tp.filter_unimportant_words(df_train, words_per_class)
    end = timer()
    timing['filtered_unimportant_words_train_set_time'] = str(timedelta(seconds=end-start))
    logging.info('Filtered unimportant words from dataset.')

    start = timer()
    filtered_test_df = tp.filter_unimportant_words(df_test, words_per_class)
    end = timer()
    timing['filtered_unimportant_words_train_set_time'] = str(timedelta(seconds=end-start))
    logging.info('Filtered unimportant words from dataset.')


    tfidf_vectorizer = TfidfVectorizer(analyzer='word')

    start = timer()
    X_train_vectorized = tfidf_vectorizer.fit_transform(filtered_train_df['Text'])
    X_test_vectorized = tfidf_vectorizer.transform(filtered_test_df['Text'])
    end = timer()
    timing['vectorized_time'] = str(timedelta(seconds=end-start))
    logging.info('Vectorized dataset.')

    start = timer()
    model.fit(X_train_vectorized, filtered_train_df['Label'])
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model,
                                X_t=X_train_vectorized,
                                y_t=filtered_train_df['Label'],
                                X_val=X_test_vectorized,
                                y_val=filtered_test_df['Label'],
                                timing=timing)

    results['n_important_words'] = n_words

    return results, words_per_class


def shap_based_method(df: pd.DataFrame, n_words: int, model: sklearn.base.BaseEstimator, split: float) -> dict:
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
    tfidf_transformer = TfidfTransformer()

    start = timer()
    X_train_vectorized_tfidf = tfidf_transformer.fit_transform(X_train_vectorized_filtered)
    X_test_vectorized_tfidf = tfidf_transformer.transform(X_test_vectorized_filtered)
    end = timer()
    timing['vectorized_time'] = str(timedelta(seconds=end-start))
    logging.info('Vectorized dataset.')

    start = timer()
    model.fit(X_train_vectorized_tfidf, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    results = record_results(model=model, 
                                X_t=X_train_vectorized_tfidf,
                                y_t=y_train,
                                X_val=X_test_vectorized_tfidf,
                                y_val=y_test,
                                timing=timing)
    results['n_words'] = n_words

    return results

def main():
    df = pd.read_csv('data/enron/enron_spam_data.csv', sep=',').sample(n=5000)
    df = df.fillna('')
    df = df.astype('str')
    
    df['Text'] = df.apply(lambda x: x['Subject'] + ', ' + x['Message'], axis=1)
    df['Label'] = np.where(df['Spam/Ham'].values == 'ham', 0, 1)
    df['Text'] = df['Text'].apply(tp.normalize_text)

    model = SVC()

    # result, words_per_class_tfidf = tfidf_based_method(df, n_words=10000, model=model, split=0.7)

    # with open('results_tfidf.json', 'w') as file:
    #     json.dump(result, file)

    # logging.info('Starting term strength testing.')
    # term_strength_extractor = filter.TermStrengthFeatureExtractor()
    # result = test_extractor(model, term_strength_extractor, df, n_words=10000, split=0.5)

    # with open('results_term_strength.json', 'w') as file:
    #     json.dump(result, file)
    
    # mi_extractor = filter.MutualInformationFeatureExtractor()
    # result = test_extractor(model, mi_extractor, df, n_words=10000, split=0.5)

    # logging.info('Starting mutual information testing.')
    # with open('results_mutual_information.json', 'w') as file:
    #     json.dump(result, file)
    
    # logging.info('Starting chi2 extractor testing.')
    # chi2_extractor = filter.Chi2FeatureExtractor()
    # result = test_extractor(model, chi2_extractor, df, n_words=10000, split=0.5)

    # with open('results_chi_2.json', 'w') as file:
    #     json.dump(result, file)

    # logging.info('Starting TRL extractor testing.')
    # trl_extractor = filter.TRLFeatureExtractor()
    # result = test_extractor(model, trl_extractor, df, n_words=10000, split=0.5)

    # with open('result_tlr.json', 'w') as file:
    #     json.dump(result, file)

    logging.info('Starting ECCD extractor testing.')
    eccd_extractor = filter.ECCDFeatureExtractor()
    result = test_extractor(model, eccd_extractor, df, n_words=10000, split=0.5)

    with open('result_eccd.json', 'w') as file:
        json.dump(result, file)


    logging.info('Starting LM extractor testing.')
    lm_extractor = filter.LinearMeasureBasedFeatureExtractor(k=50)
    result = test_extractor(model, lm_extractor, df, n_words=10000, split=0.5)

    with open('result_lm.json', 'w') as file:
        json.dump(result, file)
    # logging.info('Starting shap extractor testing.')
    # result = shap_based_method(df, n_words=10000, model=model, split=0.5)

    # with open('results_shap.json', 'w') as file:
    #     json.dump(result, file)

    

    # words_per_class_dict = {}
    # words_per_class_dict['tfidf'] = words_per_class_tfidf
    # words_per_class_dict['term_strength'] = words_per_class_term_strength
    # words_per_class_dict['mi'] = words_per_class_mi
    # words_per_class_dict['chi2'] = words_per_class_chi2
    # words_per_class_dict['shap'] = words_per_class_shap

    # with open('words_per_class_dict.pkl', 'wb') as file:
    #     pickle.dump(words_per_class_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()