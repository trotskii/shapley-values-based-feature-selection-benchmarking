import pandas as pd 
import os 
from sklearn.metrics import jaccard_score

def ensure_dir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_extractor_timings(files: dict) -> pd.DataFrame:
    methods = files.keys()
    df = pd.DataFrame(columns=methods)
    n_words_list = set()
    method_list = set()
    for method, words_dict in files.items():
        method_list.add(method)
        for n_words, info in words_dict.items():
                df.loc[n_words, method] = info['timing']['extractor_fit']
                n_words_list.add(n_words)
                
    df.index = df.index.astype('int')
    df = df.sort_index()
    df = df.apply(pd.to_timedelta)
    for col in df.columns:
        df[col] = df[col].dt.total_seconds()
    return df, list(method_list), list(n_words_list)

def get_selected_words_per_extractor_per_n_words(files: dict, vocabulary, n_words_list, method_list):
    df_dict = {}

    for n_words in n_words_list:
        df_dict[n_words] = pd.DataFrame(index=vocabulary, columns=list(method_list)).fillna(0)

    for method, words_dict in files.items():
        for n_words, info in words_dict.items():
            df_dict[n_words].loc[info['selected_vocabulary'][0], method] = 1
    return df_dict

def get_cross_jaccard_score(df):
    methods = df.columns.tolist()
    jaccard_df = pd.DataFrame(index=methods, columns=methods)
    for method_1 in methods:
        for method_2 in methods:
            jaccard_df.loc[method_1, method_2] = jaccard_score(df[method_1], df[method_2])
    return jaccard_df

def get_similarity_metrics(df_dict):
    correlations_dict = {}
    jaccard_score_dict = {}
    for n_words, df in df_dict.items():
        df_filtered = df.loc[:,(df.sum(axis=0) != 0).values] # remove lfs (or other methods) when they have no values
        correlations_dict[n_words] = df_filtered.corr() 
        jaccard_score_dict[n_words] = get_cross_jaccard_score(df_filtered)
    
    return correlations_dict, jaccard_score_dict

def compare_shap_over_n_words_set_similarity(df_dict: dict, n_words_list, method_list):
    df_comp = pd.DataFrame(columns=method_list, index=n_words_list)
    for n_words, df in df_dict.items():
        cols = df.columns
        df_comp.loc[n_words, cols] = df['shap'][cols]
    df_comp.index = df_comp.index.astype('int')
    df_comp = df_comp.sort_index()
    df_comp = df_comp.drop(columns=['shap'])
    return df_comp

def compare_methods_set_similarity(df_dict: dict, n_words_list, method_list):
    df_comp = pd.DataFrame(columns=method_list, index=n_words_list)
    for n_words, df in df_dict.items():
        cols = df.columns
        df_comp.loc[n_words, cols] = df['shap'][cols]
    df_comp.index = df_comp.index.astype('int')
    df_comp = df_comp.sort_index()
    df_comp = df_comp.drop(columns=['shap'])
    return df_comp

# def compare_performance_over_n_words(files, n_words_list, method_list, train_test, baseline=None):
#     metrics = ['precision', 'recall', 'f1-score']
#     cols = []
#     for method in method_list:
#         for metric in metrics:
#             cols.append((method, f'{metric}_mean'))
#             cols.append((method, f'{metric}_std'))
#     df_metrics = pd.DataFrame(columns=pd.MultiIndex.from_tuples(cols), index=n_words_list)

#     for method, words_dict in files.items():
#         for n_words, info in words_dict.items():
#             for metric in metrics:
#                 df_metrics.loc[n_words, (method, f'{metric}_mean')] = info[f'classification_report_{train_test}']['macro avg'][f'{metric}_mean']
#                 df_metrics.loc[n_words, (method, f'{metric}_std')] = info[f'classification_report_{train_test}']['macro avg'][f'{metric}_std']
#     if baseline is not None:
#         for n_words in n_words_list:
#             for metric in metrics:
#                 df_metrics.loc[n_words, ('baseline',f'{metric}_mean')] = baseline['macro avg'][f'{metric}_mean']
#                 df_metrics.loc[n_words, ('baseline',f'{metric}_std')] = baseline['macro avg'][f'{metric}_std']

#     df_metrics.index = df_metrics.index.astype('int')
#     df_metrics = df_metrics.sort_index()
#     return df_metrics

def compare_performance_over_n_words(files, n_words_list, method_list, train_test, baseline=None, ppfs=None):
    metrics = ['precision', 'recall', 'f1-score']
    cols = []
    for method in method_list:
        for metric in metrics:
            cols.append((method, f'{metric}_mean'))
            cols.append((method, f'{metric}_std'))
    df_metrics = pd.DataFrame(columns=pd.MultiIndex.from_tuples(cols), index=n_words_list)

    for method, words_dict in files.items():
        for n_words, info in words_dict.items():
            for metric in metrics:
                df_metrics.loc[n_words, (method, f'{metric}_mean')] = info[f'classification_report_{train_test}']['macro avg'][f'{metric}_mean']
                df_metrics.loc[n_words, (method, f'{metric}_std')] = info[f'classification_report_{train_test}']['macro avg'][f'{metric}_std']
    if baseline is not None:
        for n_words in n_words_list:
            for metric in metrics:
                df_metrics.loc[n_words, ('baseline',f'{metric}_mean')] = baseline['macro avg'][f'{metric}_mean']
                df_metrics.loc[n_words, ('baseline',f'{metric}_std')] = baseline['macro avg'][f'{metric}_std']
    
    if ppfs is not None:
        for n_words in n_words_list:
            for metric in metrics:
                df_metrics.loc[n_words, ('ppfs',f'{metric}_mean')] = ppfs['macro avg'][f'{metric}_mean']
                df_metrics.loc[n_words, ('ppfs',f'{metric}_std')] = ppfs['macro avg'][f'{metric}_std']

    df_metrics.index = df_metrics.index.astype('int')
    df_metrics = df_metrics.sort_index()
    return df_metrics