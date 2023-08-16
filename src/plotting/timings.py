from plotnine import ggplot, geom_line, geom_errorbar, aes, geom_segment, scale_x_log10, scale_y_continuous, scale_y_log10, scale_x_continuous, geom_point

from plotnine.ggplot import save_as_pdf_pages
import pandas as pd 
import numpy as np 
import os 
import json 

PATH = 'results/timings/timings.json'

def get_timings(path: str) -> pd.DataFrame:
    with open(path, 'r') as file:
        timings_json = json.load(file)
    df = pd.DataFrame.from_dict(timings_json, orient='index')

    df = df.apply(pd.to_timedelta)

    return df 

def main():
    df = get_timings(PATH)
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
                'ppfs': 'PPFS',
                'f_val': 'F-stat',
                'jmi' : 'JMI'
            }
    df = df.rename(columns=methods_renames)
    df = df.reset_index(names='n_features')
    df = df.melt(id_vars=['n_features'], value_name='Timing', var_name='Method')
    df['Timing'] = df['Timing'].dt.total_seconds()
    df['n_features'] = df['n_features'].astype('int16')

    alpha = 0.4
    p = (ggplot(df, aes(x='n_features', y='Timing', group='Method', color='Method')) + geom_line(alpha=alpha, size=2) + scale_x_continuous(name='Number of features in a dataset') + scale_y_log10(name='Runtime (s)') + geom_point(mapping=aes(x="n_features", y="Timing", shape="Method")))

    save_path = os.path.join('figures', 'timings')
    p.save(os.path.join(save_path, f'method_timings.pdf'))


if __name__ == '__main__':
    main()