import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def simulateData(N,P,K,S):
   '''
   Function1: Used to simulate the data. The data is stored into a dataframe and then using the KMeans library in sklearn the data is clustered.

   input:
   N : number of rows in the dataset
   P : number of columns in the dataset
   K : number of clusters to to be created
   S : first set of useful columns

   output: dataframe with clustered data. The column 'cluster_id' contains the id the datapoint was clustered into.
   '''
   X, _ = make_blobs(n_samples=N, centers=K, n_features=S)
   df = pd.DataFrame(X)
   kmeans = KMeans(n_clusters=K)
   y = kmeans.fit_predict(df.iloc[:,0:S])
   df['cluster_id'] = y
   return df
   

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def scoreColumns(df):
   '''
   Function 2: Used to score each column in the dataset.Higher the score, the column is better suited to build the cluster
   input: df: dataframe (returned from simulateData function)
   output: Column scores
   '''
   no_of_columns = len(df.columns)
   model_features = SelectKBest(score_func=chi2, k=no_of_columns)
   fit = model_features.fit(df.iloc[:,0:no_of_columns-1],df.iloc[:,no_of_columns])
   print(fit.scores_)

 
 
   
