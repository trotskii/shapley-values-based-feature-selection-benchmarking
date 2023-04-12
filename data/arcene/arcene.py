import pandas as pd 
import numpy as np 
import os
from scipy.io.arff import loadarff

def main():
    path = os.path.realpath(__file__)
    path = os.path.split(path)[0]
    arff = loadarff(os.path.join(path, 'arcene.arff'))
    df_data = pd.DataFrame(arff[0])
    df_data['Class'] = df_data['Class'].astype(int)
    df_data.to_csv(os.path.join(path, 'arcene.csv'), sep=';')

    print(df_data.groupby('Class').count())
    

if __name__ == '__main__':
    main()