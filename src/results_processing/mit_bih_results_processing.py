import json 
import pandas as pd
import os
import numpy as np
import sklearn.feature_extraction.text as ft 
import src.preprocessing.text_preprocessing as tp
from src.results_processing.results_processing_functions import *

FOLDER = 'ecg_mit_half'
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
    df = pd.read_csv('data/mitdb_data/mitdb_ecg.csv', sep=';')
    vocabulary = df.columns.tolist()

    df, method_list, n_words_list = get_extractor_timings(files)
    ensure_dir_path('csv_results_outputs/timings')
    df.to_csv('csv_results_outputs/timings/mit_bih_n_word_timings_new.csv', sep=';')

    df_dict = get_selected_words_per_extractor_per_n_words(files, vocabulary, n_words_list, method_list)
    _, jaccard_score_dict = get_similarity_metrics(df_dict)

    with open('results/mit_bih_baseline.json', 'r') as file:
        baseline_arcene = json.load(file)

    for TEST_TRAIN in ['test', 'train']:
        ensure_dir_path('csv_results_outputs/jaccard/')
        shap_jaccard = compare_shap_over_n_words_set_similarity(jaccard_score_dict, n_words_list, method_list)
        shap_jaccard.to_csv(f'csv_results_outputs/jaccard/mit_bih_jaccard_{TEST_TRAIN}.csv', sep=';')
        
        ensure_dir_path('csv_results_outputs/metrics/mit_bih/')
        df_metrics = compare_performance_over_n_words(files, n_words_list, method_list, TEST_TRAIN, baseline=baseline_arcene[f'classification_report_{TEST_TRAIN}'])
        df_metrics.to_csv(f'csv_results_outputs/metrics/mit_bih/mit_bih_performance_{TEST_TRAIN}.csv', sep=';')


if __name__ == '__main__':
    main()