# Import libraries
import EDA_sleeplog as es
import EDA_timeseries as et

import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Classifier as CLF
import pickle
import time as time

# Main program
if __name__ == '__main__':
    _ = es.ChangeDir()

    # Load data
    file = './test_series.parquet'
    df = pl.scan_parquet('./test_series.parquet').with_columns(
        (
            (pl.col('step').cast(pl.UInt32)), 
            (pl.col('anglez').round(0).cast(pl.Int8))
        )
    ).select(['series_id', 'step', 'timestamp', 'anglez']).collect().to_pandas()
    sid = df.series_id.unique()

    dfAvg = et.RollingAvg(df, sid, 2, 1, 'test')
    X = dfAvg.iloc[:, 2:]

    scaler = MinMaxScaler()
    model = scaler.fit(X)
    X = model.transform(X)

    # Load and deploy the model
    clf = pickle.load(open('./model2', 'rb'))
    y_pred = clf.predict(X).tolist()

    # Extract windows that are different from the initial state
    dfAvg['prediction'] = pd.Series(y_pred)
    ans = {}

    for s in sid:
        
        window = []

        dfRaw = df[df['series_id'] == s].reset_index()
        tg = dfAvg[dfAvg.sid == s].reset_index()
        iniS = tg.iloc[0,5] # initial state
        predS = tg.iloc[:,5]

        for i in range(len(predS)):
            if predS[i] != iniS:
                window.append(tg.iloc[i,2])

        # Determine the state of the period
        cont = 0
        if len(window) >= 359 :
            for i in range(len(window)):
                if i != len(window)-1:
                    if window[i+1] - window[i] == 1:
                        cont += 1
                    elif len(window[i:]) >= 359:
                        cont = 0
                    elif cont >= 359:
                        



        # elif len(window) == :

    submission = pd.DataFrame()
pass