from plotnine import ggplot, geom_line, geom_errorbar, aes, stat_smooth, facet_wrap, labs, lims, scale_x_log10, scale_y_continuous, scale_y_log10
from plotnine.ggplot import save_as_pdf_pages
import pandas as pd 
import numpy as np 
import os 


def _ensure_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


BASE_FOLDER = 'csv_results_outputs/metrics'
datasets = os.listdir(BASE_FOLDER)
files_dict = {dataset: [os.path.join(BASE_FOLDER, dataset, f) for f in os.listdir(os.path.join(BASE_FOLDER, dataset))] for dataset in datasets}
df_dict = {}
for key, item in files_dict.items():
    if key not in df_dict:
        df_dict[key] = {}
    for i in item:
        train_test = i[:-4].split('_')[-1]
        df_dict[key][train_test] = pd.read_csv(i, sep=';', header=[0,1], index_col=0)


for dataset_name, dataset_dict in df_dict.items():
    for train_test, df in dataset_dict.items():

        index = df.index
        df = df.stack(0).reset_index(1).reset_index().rename(columns={'index': 'n_features', 'level_1': 'method_name'})

        for metric in ['f1-score', 'precision', 'recall']:
            sel_cols = [col for col in df.columns if metric in col]
            sel_cols.extend(['n_features', 'method_name'])
            df_filtered = df[sel_cols].rename(columns={f'{metric}_std': 'std', f'{metric}_mean':'mean'})
            # df_filtered = df_filtered.loc[df_filtered['method_name']!='trl']
            alpha = 0.4

            if 'remove' in dataset_name:
                x_label = 'Number of the removed features'
            else:
                x_label = 'Number of the kept features'
            
            methods_renames = {
                'shap': 'SHAP',
                'term_strength': 'TS',
                'trl': 'TRL',
                'eccd': 'ECCD',
                'mutual_information': 'MI',
                'chi2': 'chi2',
                'tfidf': 'TF-IDF',
                'linear_measure_5': 'LM',
                'lfs': 'LFS',
                'baseline': 'Baseline',
                'ppfs': 'PPFS',
                'f_val': 'F-stat'
            }
            df_filtered['method_name'] = df_filtered['method_name'].map(methods_renames)
            df_filtered = df_filtered.rename(columns={'method_name': 'Method'})
            p = (ggplot(df_filtered, aes(x='n_features', y='mean', group='Method', color='Method')) + geom_line(alpha=alpha, size=2) + geom_errorbar(aes(x='n_features', ymin='mean-std', ymax='mean+std', group='Method'), linetype='solid', alpha=alpha, width=0.05, size=2) + scale_x_log10(name=x_label) + scale_y_continuous(name=metric.capitalize()))  

            save_path = os.path.join('figures', 'metrics', f'{dataset_name}')
            _ensure_dir_path(save_path)

            p.save(os.path.join(save_path, f'{dataset_name}_{metric}_{train_test}.pdf'))

BASE_FOLDER = 'csv_results_outputs/timings'
datasets = os.listdir(BASE_FOLDER)
files_dict = {dataset: [os.path.join(BASE_FOLDER, dataset, f) for f in os.listdir(os.path.join(BASE_FOLDER, dataset))] for dataset in datasets}
df_dict = {}
for key, item in files_dict.items():
    if key not in df_dict:
        df_dict[key] = {}
    for i in item:
        train_test = i[:-4].split('_')[-1]
        df_dict[key][train_test] = pd.read_csv(i, sep=';', header=[0,1], index_col=0)

for dataset_name, dataset_dict in df_dict.items():
    for train_test, df in dataset_dict.items():
        df = df.reset_index().rename(columns={'index': 'n_features'})
        df = pd.melt(df, id_vars='n_features', var_name='Method')

        alpha = 0.4

        methods_renames = {
            'shap': 'SHAP',
            'term_strength': 'TS',
            'trl': 'TRL',
            'eccd': 'ECCD',
            'mutual_information': 'MI',
            'chi2': 'chi2',
            'tfidf': 'TF-IDF',
            'linear_measure_5': 'LM',
            'lfs': 'LFS',
            'baseline': 'Baseline',
            'ppfs': 'PPFS',
            'f_val': 'F-stat'
        }
        df['Method'] = df['Method'].map(methods_renames)
        p = (ggplot(df, aes(x='n_features', y='value', group='Method', color='Method')) + geom_line(alpha=alpha, size=2) + scale_x_log10(name='Number of kept features') + scale_y_log10(name='Runtime (s)'))  

        save_path = os.path.join('figures', 'timings', f'{dataset_name}')
        _ensure_dir_path(save_path)

        p.save(os.path.join(save_path, f'{dataset_name}_timing_{train_test}.pdf'))
