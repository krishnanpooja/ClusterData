'''
Function1: Used to simulate the data. The data is stored into a dataframe and then using the KMeans library in sklearn the data is clustered.

input:
N : number of rows in the dataset
P : number of columns in the dataset
K : number of clusters to to be created
S : first set of useful columns

output: dataframe with clustered data. The column 'cluster_id' contains the id the datapoint was clustered into.
'''

import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def simulateData(N,P,K,S):
   X, _ = make_blobs(n_samples=N, centers=K, n_features=S)
   df = pd.DataFrame(X)
   kmeans = KMeans(n_clusters=K)
   y = kmeans.fit_predict(df.iloc[:,0:S])
   df['cluster_id'] = y
   return df
   
 
 
   
