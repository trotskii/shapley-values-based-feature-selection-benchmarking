import pandas as pd 
import numpy as np 
import sklearn 
from timeit import default_timer as timer
from datetime import timedelta
import logging
import json
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


import src.preprocessing.text_preprocessing as tp 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def tfidf_based_method(df: pd.DataFrame, n_words: int, model: sklearn.base.BaseEstimator, split: float) -> dict:
    """
    Train passed model with corresponding parameters according to tfidf feature selection.
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


    start = timer()
    words_per_class = tp.get_n_most_important_words_cftfidf(df, n_words=n_words)
    end = timer()
    timing['n_important_words_time'] = str(timedelta(seconds=end-start))
    logging.info('Got important words per class.')
    

    start = timer()
    filtered_df = tp.filter_unimportant_words(df, words_per_class)
    end = timer()
    timing['filtered_unimportant_words_time'] = str(timedelta(seconds=end-start))
    logging.info('Filtered unimportant words from dataset.')

    X_train, X_test, y_train, y_test = train_test_split(filtered_df['Text'].values, filtered_df['Label'].values, test_size=split, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word')

    start = timer()
    X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
    X_test_vectorized = tfidf_vectorizer.transform(X_test)
    end = timer()
    timing['vectorized_time'] = str(timedelta(seconds=end-start))
    logging.info('Vectorized dataset.')

    start = timer()
    model.fit(X_train_vectorized, y_train)
    end = timer()    
    timing['model_training_time'] = str(timedelta(seconds=end-start))
    logging.info('Model training finished.')

    predictions_train = model.predict(X_train_vectorized)

    start = timer()
    predictions_test = model.predict(X_test_vectorized)
    end = timer()
    timing['model_inference_time'] = str(timedelta(seconds=end-start))
    logging.info('Model inference finished.')

    report_train = classification_report(y_train, predictions_train, output_dict=True)
    report_test = classification_report(y_test, predictions_test, output_dict=True)

    cm_train = confusion_matrix(y_train, predictions_train, normalize='true')
    cm_test = confusion_matrix(y_test, predictions_test, normalize='true')

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

    result = tfidf_based_method(df, n_words=10000, model=model, split=0.7)

    with open('results.json', 'w') as file:
        json.dump(result, file)

if __name__ == '__main__':
    main()