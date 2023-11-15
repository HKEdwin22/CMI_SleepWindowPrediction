# Import libraries
import EDA_sleeplog as es
import pandas as pd
from sklearn.model_selection import train_test_split
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
   clf.fit(X_train, y_train)
   pickle.dump(clf, open('./model1', 'wb'))
   y_pred = clf.predict(X_test)
   print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}')

   

# Main program
if __name__ == '__main__':

   startExe = time.time()
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   UsrAns = False
   if UsrAns:
      TrainModel('./training set 2.csv')

   # Load data
   file = './test_series.parquet'
   df = pl.scan_parquet(file).collect().pandas()

   
   
   # Load and deploy the model
   clf = pickle.load(open('./model1', 'rb'))

   endExe = time.time()
   print(f'Execution time : {(endExe-startExe):.2f} seconds')
   pass
