import pandas as pd 
import numpy as np 
import scipy
import sklearn 
from timeit import default_timer as timer
from datetime import timedelta
import logging
import json
import sys
from functools import partial
import mifs
from PyImpetus import PPIMBC
from sklearn.tree import DecisionTreeClassifier
from src.preprocessing.ctfidf import CTFIDFVectorizer

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from numpy.typing import ArrayLike

import src.preprocessing.text_preprocessing as tp 
import src.preprocessing.feature_extraction.text.filtering as filter 
import src.preprocessing.feature_extraction.text.wrapping as wrapping

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


def create_datset(n_features: int, n_samples: int, is_discrete: bool):
    if is_discrete:
        X = np.random.choice(a=5, size=(n_samples, n_features))
    else:
        X = np.random.uniform(low=0, high=1, size=(n_samples, n_features))
    Y = np.random.choice(a=2, size=(n_samples, ))

    return X, Y 

def time_extractor(extractor, X:np.ndarray, Y:np.ndarray):
    start = timer()
    extractor.fit(X, Y)
    end = timer()
    fitting_time = timedelta(seconds=end-start)

    return str(fitting_time)

def main():
    feature_extractors = {}
    timings = {}
    feature_extractors['term_strength'] = filter.TermStrengthFeatureExtractor()
    feature_extractors['mutual_information'] = filter.MutualInformationFeatureExtractor()
    feature_extractors['chi2'] = filter.Chi2FeatureExtractor()
    feature_extractors['trl'] = filter.TRLFeatureExtractor()
    feature_extractors['eccd'] = filter.ECCDFeatureExtractor()
    feature_extractors['linear_measure_5'] = filter.LinearMeasureBasedFeatureExtractor(k=5)
    feature_extractors['f_val'] = filter.FValFeatureExtractor()
    feature_extractors['shap'] = wrapping.ShapFeatureExtractor()
    
    n_features = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    discrete_methods = ['term_strength', 'chi2', 'linear_measure_5', 'ctfidf']
    for n_f in n_features:
        timings[n_f] = {}
        for name, extractor in feature_extractors.items():
            if name in discrete_methods:
                X, Y = create_datset(n_features=n_f, n_samples=5000, is_discrete=True)
            else:
                X, Y = create_datset(n_features=n_f, n_samples=5000, is_discrete=False)

            time = time_extractor(extractor, X, Y)
            print(f'Tested {name} on {n_f} features: {time}')
            timings[n_f][name] = time

            with open(f'results/timings/timings.json', 'w') as file:
                json.dump(timings, file) 
    for n_f in n_features:

        jmi = mifs.MutualInformationFeatureSelector(method='JMI', k=10, n_features=int(n_f/10)+1, verbose=0, n_jobs=-1)
        lfs = wrapping.LinearForwardSearch(estimator=SVC(), ranker=filter.MutualInformationFeatureExtractor(), vocabulary=None)
        ppfs = PPIMBC(DecisionTreeClassifier(random_state=27), p_val_thresh=0.05, num_simul=30, cv=5, n_jobs=-1, verbose=0)
        ctfidf = CTFIDFVectorizer()

        X, Y = create_datset(n_features=n_f, n_samples=5000, is_discrete=False)
        
        time = time_extractor(jmi, X, Y)
        timings[n_f]['jmi'] = time
        print(f'Tested JMI at {n_f}: {time}')

        with open(f'results/timings/timings.json', 'w') as file:
            json.dump(timings, file) 

        # ### lfs start
        # start = timer()
        # lfs.fit(X=X, y=Y, k=50, n_words=int(n_f/10)+1)
        # end = timer()
        # ### lfs end
        # time = str(timedelta(seconds=end-start))
        # timings[n_f]['lfs'] = time
        # print(f'Tested lfs at {n_f}: {time}')


        with open(f'results/timings/timings.json', 'w') as file:
            json.dump(timings, file) 

        ### ctfidf start
        X, Y = create_datset(n_features=n_f, n_samples=5000, is_discrete=True)

        start = timer()
        ctfidf.fit_transform(X, n_samples=5000)
        end = timer()
        ### ctfidf end
        time = str(timedelta(seconds=end-start))
        timings[n_f]['ctfidf'] = time
        print(f'Tested ctfidf at {n_f}: {time}')

        with open(f'results/timings/timings.json', 'w') as file:
            json.dump(timings, file)


        X, Y = create_datset(n_features=n_f, n_samples=5000, is_discrete=False)

        ### ppfs start
        start = timer()
        ppfs.fit_transform(X, Y)
        end = timer()
        ### ppfs end
        time = str(timedelta(seconds=end-start))
        timings[n_f]['ppfs'] = time
        print(f'Tested ppfs at {n_f}: {time}')

        with open(f'results/timings/timings.json', 'w') as file:
            json.dump(timings, file)





if __name__ == '__main__':
    main()