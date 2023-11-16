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
    seriesID, step, event = [], [], []

    for s in sid:
        
        window = []

        dfRaw = df[df['series_id'] == s].reset_index()
        tg = dfAvg[dfAvg.sid == s].reset_index()
        iniS = tg.iloc[0,5] # initial state
        predS = tg.iloc[:,5]

        for i in range(len(predS)):
            if predS[i] != iniS:
                window.append(tg.iloc[i,2])

        # Identify the window of changes
        cont = 0
        chgWin = 0
        chgRcd = []
        if len(window) >= 359 :
            for i in range(len(window)):
                if i != len(window)-1:
                    if window[i+1] - window[i] == 1:
                        cont += 1
                    else:
                        if cont >= 359:
                            chgRcd.append(chgWin)
                            if len(window[i:]) >= 359:
                                chgWin = i+1
                                cont = 0
                            elif len(window[i:]) < 359:
                                break
                        
        if chgRcd != []:
            for i in chgRcd:
                state0 = tg[tg['window'] == i-1].prediction
                state1 = tg[tg['window'] == i].prediction
                if state0 == state1:
                    print('Error: pending modifications')
                else:
                    seriesID.append(s)
                    step.append((tg[tg['window'] == i]).step)
                    event.append('wakeup' if state1 == 'awake' else 'onset')

    # Save results into the submission file
    submission = {'row_id': [i for i in range(len(seriesID))], 
                  'series_id': seriesID,
                  'step': step,
                  'event': event
                  }
    submission = pd.DataFrame(submission, index=False)
    submission.to_csv('./submission.csv')
    
pass