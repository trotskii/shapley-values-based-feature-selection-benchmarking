import pandas as pd 
import numpy as np 
import scipy
import sklearn 
from timeit import default_timer as timer
from datetime import timedelta
import logging
import json
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC


import src.preprocessing.text_preprocessing as tp 
import src.preprocessing.feature_extraction.text.filtering as filter 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def term_strength_based_method(df: pd.DataFrame, n_words: int, model: sklearn.base.BaseEstimator, split: float) -> dict:
    """
    Train passed model according to term strength feature selection.
    """

    results = {}
    timing = {}

    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Label'], test_size=split)
    df_train = pd.DataFrame(X_train)
    df_train['Label'] = y_train 

    df_test = pd.DataFrame(X_test)
    df_test['Label'] = y_test

    count_vectorizer = CountVectorizer(binary=True)
    count_vectorizer.fit(X_train)

    X_train_vectorized = count_vectorizer.transform(X_train)

    term_strength_extractor = filter.TermStrengthFeatureExtractor()
    start = timer()
    term_strength_extractor.fit(X_train_vectorized, y_train)
    end = timer()
    timing['term_strength_fit'] = str(timedelta(seconds=end-start))
    logging.info('Fit term strength extractor.')

    words_per_class = term_strength_extractor.get_n_strongest_terms_words(n_words, 
                                    vocabulary=count_vectorizer.get_feature_names_out())

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

    predictions_train = model.predict(X_train_vectorized)

    start = timer()
    predictions_test = model.predict(X_test_vectorized)
    end = timer()
    timing['model_inference_time'] = str(timedelta(seconds=end-start))
    logging.info('Model inference finished.')

    report_train = classification_report(filtered_train_df['Label'], predictions_train, output_dict=True)
    report_test = classification_report(filtered_test_df['Label'], predictions_test, output_dict=True)

    cm_train = confusion_matrix(filtered_train_df['Label'], predictions_train, normalize='true')
    cm_test = confusion_matrix(filtered_test_df['Label'], predictions_test, normalize='true')

    results['timing'] = timing
    results['training_data_samples'] = X_train.shape[0]
    results['test_data_samples'] = X_test.shape[0]
    results['classification_report_train'] = report_train
    results['classification_report_test'] = report_test
    results['confustion_matrix_train'] = cm_train.tolist()
    results['confusion_matrix_test'] = cm_test.tolist()
    results['model_type'] = type(model).__name__
    results['model_params'] = model.get_params()
    results['n_important_words'] = n_words

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
    results = {}
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

    predictions_train = model.predict(X_train_vectorized)

    start = timer()
    predictions_test = model.predict(X_test_vectorized)
    end = timer()
    timing['model_inference_time'] = str(timedelta(seconds=end-start))
    logging.info('Model inference finished.')

    report_train = classification_report(filtered_train_df['Label'], predictions_train, output_dict=True)
    report_test = classification_report(filtered_test_df['Label'], predictions_test, output_dict=True)

    cm_train = confusion_matrix(filtered_train_df['Label'], predictions_train, normalize='true')
    cm_test = confusion_matrix(filtered_test_df['Label'], predictions_test, normalize='true')

    results['timing'] = timing
    results['training_data_samples'] = X_train.shape[0]
    results['test_data_samples'] = X_test.shape[0]
    results['classification_report_train'] = report_train
    results['classification_report_test'] = report_test
    results['confustion_matrix_train'] = cm_train.tolist()
    results['confusion_matrix_test'] = cm_test.tolist()
    results['model_type'] = type(model).__name__
    results['model_params'] = model.get_params()
    results['n_important_words'] = n_words

    return results


   




def main():
    df = pd.read_csv('data/enron/enron_spam_data.csv', sep=',')
    df = df.fillna('')
    df = df.astype('str')
    
    df['Text'] = df.apply(lambda x: x['Subject'] + ', ' + x['Message'], axis=1)
    df['Label'] = np.where(df['Spam/Ham'].values == 'ham', 0, 1)
    df['Text'] = df['Text'].apply(tp.normalize_text)

    model = SVC()

    # result = tfidf_based_method(df, n_words=10000, model=model, split=0.7)

    # with open('results_tfidf.json', 'w') as file:
    #     json.dump(result, file)

    result = term_strength_based_method(df, n_words=10000, model=model, split=0.7)

    with open('results_term_strength.json', 'w') as file:
        json.dump(result, file)

if __name__ == '__main__':
    main()