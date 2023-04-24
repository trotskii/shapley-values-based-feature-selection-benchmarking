import json 
import pandas as pd
import os
import numpy as np
import sklearn.feature_extraction.text as ft 
import src.preprocessing.text_preprocessing as tp
from src.results_processing.results_processing_functions import *

FOLDER = 'arcene'
base_path = 'results/'
path = f'{base_path}{FOLDER}/'
full_paths = [f'{path}{file}' for file in os.listdir(path)]

def main():
    files = {}
    for p in full_paths:
        name = p.split('/')[-1]
        if name == 'results_ppfs.json':
            continue # handle separately together with baseline
        split = name.split('_')
        method = '_'.join(split[1:-1])
        n_words = split[-1].split('.')[0]
        with open(p, 'r') as file:
            if method not in files:
                files[method] = {}
            files[method][n_words] = json.load(file)
    
    df = pd.read_csv('data/arcene/arcene.csv', sep=';', index_col=0)
    df = df.fillna('')
    df = df.astype('float')
    df['Class'] = df['Class'].astype('category')
    df['Class'] = df['Class'].cat.codes


    vocabulary = df.columns[:-1]
    
    df, method_list, n_words_list = get_extractor_timings(files)
    ensure_dir_path('csv_results_outputs/timings/arcene')
    df.to_csv('csv_results_outputs/timings/arcene/arcene_n_word_timings.csv', sep=';')

    df_dict = get_selected_words_per_extractor_per_n_words(files, vocabulary, n_words_list, method_list)
    _, jaccard_score_dict = get_similarity_metrics(df_dict)

    with open('results/arcene_baseline.json', 'r') as file:
        baseline_arcene = json.load(file)

    with open('results/arcene/results_ppfs.json', 'r') as file:
        ppfs_arcene = json.load(file)

    for TEST_TRAIN in ['test', 'train']:
        ensure_dir_path('csv_results_outputs/jaccard/')
        shap_jaccard = compare_shap_over_n_words_set_similarity(jaccard_score_dict, n_words_list, method_list)
        shap_jaccard.to_csv(f'csv_results_outputs/jaccard/arcene_jaccard_{TEST_TRAIN}.csv', sep=';')
        
        ensure_dir_path('csv_results_outputs/metrics/arcene/')
        df_metrics = compare_performance_over_n_words(files, n_words_list, method_list, TEST_TRAIN, baseline=baseline_arcene[f'classification_report_{TEST_TRAIN}'], ppfs=ppfs_arcene[f'classification_report_{TEST_TRAIN}'])
        df_metrics.to_csv(f'csv_results_outputs/metrics/arcene/arcene_performance_{TEST_TRAIN}.csv', sep=';')


if __name__ == '__main__':
    main()