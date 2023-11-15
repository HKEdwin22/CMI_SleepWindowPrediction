# Import libraries
import EDA_sleeplog as es
import EDA_timeseries as et
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import pickle
import time


# Functions
def TrainModel(f):
   '''
   Train models
   f : input file name (.csv)
   '''

   # Load and prepare the features and outputs
   df = pd.read_csv(f)
   X = df.iloc[:, 2:4]
   y = df.iloc[:, 4]
   
   # Normalise the data
   scaler = MinMaxScaler()
   model = scaler.fit(X)
   X = model.transform(X)

   # Prepare the training and evaluation sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = 0.2, random_state=13)

   # Build and evaluate the model
   clf = GaussianNB()
   
   # Cross validation
   scores = cross_validate(clf, X, y, cv=5, return_train_score=True)
   print(f'Train Accuracy: {scores["train_score"].mean():.4f}\tTest Accuracy: {scores["test_score"].mean():.4f}')

   clf.fit(X_train, y_train)
   pickle.dump(clf, open('./model2', 'wb'))
   y_pred = clf.predict(X_test)
   print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}')


   

# Main program
if __name__ == '__main__':

   startExe = time.time()
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   UsrAns = True
   if UsrAns:
      TrainModel('./training set 2.csv')

   # Load data
   file = './test_series.parquet'
   df = pl.scan_parquet(file).collect().to_pandas()
   sid = df.series_id.unique()

   dfAvg = et.RollingAvg(df, sid, 6, 6, 'test')
   X = dfAvg.iloc[:, 2:]

   scaler = MinMaxScaler()
   model = scaler.fit(X)
   X = model.transform(X)

   # Load and deploy the model
   clf = pickle.load(open('./model2', 'rb'))

   y_pred = clf.predict(X).tolist()
   idx = []
   for i in range(len(y_pred)):
      if y_pred[i] == 'awake':
         idx.append(i)

   endExe = time.time()
   print(f'Execution time : {(endExe-startExe):.2f} seconds')
   pass