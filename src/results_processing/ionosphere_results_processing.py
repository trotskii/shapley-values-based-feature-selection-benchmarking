import json 
import pandas as pd
import os
import numpy as np
import sklearn.feature_extraction.text as ft 
import src.preprocessing.text_preprocessing as tp
from src.results_processing.results_processing_functions import *

FOLDER = 'ionosphere'
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
    
    df = pd.read_csv('data/ionosphere/ionosphere.csv', sep=',', index_col=0, header=None)
    cols = [f'var_{i}' for i in range(df.shape[1]-1)]
    cols.append('label')
    df.columns = cols

    vocabulary = df.columns[:-1]
    df, method_list, n_words_list = get_extractor_timings(files)
    ensure_dir_path('csv_results_outputs/timings/ionosphere')
    df.to_csv('csv_results_outputs/timings/ionosphere/ionosphere_n_word_timings.csv', sep=';')



    with open('results/ionosphere_baseline.json', 'r') as file:
        baseline_arcene = json.load(file)

    with open('results/ionosphere/results_ppfs.json', 'r') as file:
        ppfs_arcene = json.load(file)

    for TEST_TRAIN in ['test', 'train']:
        ensure_dir_path('csv_results_outputs/metrics/ionosphere/')
        df_metrics = compare_performance_over_n_words(files, n_words_list, method_list, TEST_TRAIN, baseline=baseline_arcene[f'classification_report_{TEST_TRAIN}'], ppfs=ppfs_arcene[f'classification_report_{TEST_TRAIN}'], ppfs_n_words=np.mean(ppfs_arcene['n_words']))
        df_metrics.to_csv(f'csv_results_outputs/metrics/ionosphere/ionosphere_performance_{TEST_TRAIN}.csv', sep=';')


if __name__ == '__main__':
    main()