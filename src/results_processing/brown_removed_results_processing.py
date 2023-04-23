import json 
import pandas as pd
import os
import numpy as np
import sklearn.feature_extraction.text as ft 
import src.preprocessing.text_preprocessing as tp
from src.results_processing.results_processing_functions import *

FOLDER = 'brown_remove'
base_path = 'results/'

def main():
    path = f'{base_path}{FOLDER}/'
    full_paths = [f'{path}{file}' for file in os.listdir(path)]
    files = {}
    for p in full_paths:
        name = p.split('/')[-1]
        split = name.split('_')
        method = '_'.join(split[1:-1])
        n_words = split[-1].split('.')[0]
        with open(p, 'r') as file:
            if method not in files:
                files[method] = {}
            files[method][n_words] = json.load(file)
    
    df = pd.read_csv('data/brown_corpus/brown_corpus.csv', sep=';')
    df = df.fillna('')
    df = df.astype('str')
    df['Label'] = df['Label'].astype('category')
    df['Label'] = df['Label'].cat.codes
    df['Text'] = df['Text'].apply(tp.normalize_text)

    count_vectorizer = ft.CountVectorizer()
    count_vectorizer.fit(df['Text'])
    vocabulary = count_vectorizer.get_feature_names_out()

    df, method_list, n_words_list = get_extractor_timings(files)
    ensure_dir_path('csv_results_outputs/timings')

    df_dict = get_selected_words_per_extractor_per_n_words(files, vocabulary, n_words_list, method_list)
    _, jaccard_score_dict = get_similarity_metrics(df_dict)

    with open('results/brown_baseline.json', 'r') as file:
            baseline = json.load(file)

    for TEST_TRAIN in ['test', 'train']:
        ensure_dir_path('csv_results_outputs/metrics/brown_removed/')
        df_metrics = compare_performance_over_n_words(files, n_words_list, method_list, TEST_TRAIN, baseline=baseline[f'classification_report_{TEST_TRAIN}'])
        df_metrics.to_csv(f'csv_results_outputs/metrics/brown_removed/brown_removed_performance_{TEST_TRAIN}.csv', sep=';')


if __name__ == '__main__':
    main()