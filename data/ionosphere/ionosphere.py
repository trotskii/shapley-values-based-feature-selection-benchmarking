import pandas as pd 

df = pd.read_csv('ionosphere.csv', sep=',', header=None)
print(df.columns)
print(df.groupby(34).count())