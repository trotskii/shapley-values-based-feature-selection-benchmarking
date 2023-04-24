import json 
import pandas as pd
import os
import numpy as np
import sklearn.feature_extraction.text as ft 
import src.preprocessing.text_preprocessing as tp
from src.results_processing.results_processing_functions import *

FOLDER = 'enron'
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
    
    df = pd.read_csv('data/enron/enron_spam_data.csv', sep=',')
    df = df.fillna('')
    df = df.astype('str')
    df['Text'] = df.apply(lambda x: x['Subject'] + ', ' + x['Message'], axis=1)
    df['Label'] = np.where(df['Spam/Ham'].values == 'ham', 0, 1)
    df['Text'] = df['Text'].apply(tp.normalize_text)

    count_vectorizer = ft.CountVectorizer()
    count_vectorizer.fit(df['Text'])
    vocabulary = count_vectorizer.get_feature_names_out()

    df, method_list, n_words_list = get_extractor_timings(files)
    ensure_dir_path('csv_results_outputs/timings/enron')
    df.to_csv('csv_results_outputs/timings/enron/enron_n_word_timings_new.csv', sep=';')

    df_dict = get_selected_words_per_extractor_per_n_words(files, vocabulary, n_words_list, method_list)
    _, jaccard_score_dict = get_similarity_metrics(df_dict)

    with open('results/enron_baseline.json', 'r') as file:
            baseline_enron = json.load(file)

    for TEST_TRAIN in ['test', 'train']:
        ensure_dir_path('csv_results_outputs/jaccard/')
        shap_jaccard = compare_shap_over_n_words_set_similarity(jaccard_score_dict, n_words_list, method_list)
        shap_jaccard.to_csv(f'csv_results_outputs/jaccard/enron_jaccard_{TEST_TRAIN}.csv', sep=';')
        
        ensure_dir_path('csv_results_outputs/metrics/enron/')
        df_metrics = compare_performance_over_n_words(files, n_words_list, method_list, TEST_TRAIN, baseline=baseline_enron[f'classification_report_{TEST_TRAIN}'])
        df_metrics.to_csv(f'csv_results_outputs/metrics/enron/enron_performance_{TEST_TRAIN}.csv', sep=';')


if __name__ == '__main__':
    main()