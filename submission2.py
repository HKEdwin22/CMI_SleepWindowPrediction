# Import libraries
import EDA_sleeplog as es
import EDA_timeseries as et

import polars as pl
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Classifier as CLF
import pickle
import time as time
from tqdm import tqdm

# Main program
if __name__ == '__main__':

    _ = es.ChangeDir()

    startExe = time.time()

    # Load data
    file = './test_series.parquet'
    df = pl.scan_parquet(file).with_columns(
        (
            (pl.col('step').cast(pl.UInt32)), 
            (pl.col('anglez').round(0).cast(pl.Int8))
        )
    ).select(['series_id', 'step', 'timestamp', 'anglez']).collect().to_pandas()
    sid = df.series_id.unique()

    print('-'*20 + 'Compute rolling mean and standard deviation' + '-'*20)
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

    for s in tqdm(sid):
        
        window = []

        dfRaw = df[df['series_id'] == s].reset_index()
        tg = dfAvg[dfAvg.sid == s].reset_index()
        iniS = tg.iloc[0,5] # initial state
        predS = tg.iloc[:,5]

        # Identify windows that have changed state
        for i in range(len(predS)):
            if predS[i] != iniS:
                window.append(tg.iloc[i,2])
                iniS = tg.iloc[i,5]

        # Identify changed states that last for 360 steps (30mins)
        chWinConfirmed = []
        for i in range(len(window)):
            if i+1 != len(window):
                if window[i+1] - window[i] >= 360:
                    chWinConfirmed.append(window[i])
            elif tg.iloc[-1,2] - window[i] >=360:
                chWinConfirmed.append(window[i])

        # Select the exact step as the final result
        if chWinConfirmed != []:
            extStep = {}
            for i in chWinConfirmed:
                angle1 = df[(df.series_id == s) & (df.step == i)]
                angle2 = df[(df.series_id == s) & (df.step == i+1)]
                std = tg.iloc[i-2, 4]

                if angle1>=-3*std and angle1<=3*std:
                    if angle2<=-3*std or angle2>=3*std:
                        extStep[i] = i+1
                else:
                    extStep[i] = i

            for w, s in extStep:
                seriesID.append(s)
                step.append(s)
                if tg.iloc[w-1,5] == 'sleep':
                    event.append('onset')
                else:
                    event.append('wakeup')

    # Save results into the submission file
    submission = {'row_id': [i for i in range(len(seriesID))], 
                  'series_id': seriesID,
                  'step': step,
                  'event': event
                  }
    submission = pd.DataFrame(submission, index=None)
    submission.to_csv('./submission.csv')

endExe = time.time()
print(f'Execution time : {(endExe-startExe):.2f} seconds')

pass