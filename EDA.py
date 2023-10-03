'''
This script is used for EDA of the project.
'''

# Import libraries
import os
import polars as pl
import pandas as pd
import numpy as np
import dtale as dt
import seaborn as sns
import matplotlib.pyplot as plt


def DescriptiveStat(x, t):
    '''
    Generate an overall statistical description of the dataset
    x : input dataset
    t : file type (csv or parquet)
    '''

    if t == 'csv':
        df = pd.read_csv(x)
    
    # d = dt.show(df)
    # print(d._main_url)
    print(df.info())
    print(df.nunique())
    print(df.describe())

def ReadParquet(x):
    '''
    Load a parquet file and return a panda dataframe
    x : input dataset
    '''
    y = pl.scan_parquet(x, n_rows=15000)
                # .with_columns(
                #     (
                #         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z")),
                #         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.year().alias("year")),
                #         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.month().alias("month")),
                #         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.day().alias("day")),
                #         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.hour().alias("hour")),
                #     )
                # )
    y = y.collect()
    
    return y.to_pandas()


# Main program
if __name__ == '__main__':

    dir_old = os.getcwd()    
    os.chdir('../../')      # work under "My Documents"
    dir_MyDoc = os.getcwd()
    target_path = 'DSAI\Kaggle_Competitions\CMI_Detect Sleep States\RawData'
    os.chdir(os.path.join(dir_MyDoc, target_path))
    
    # Load train_events.csv
    file = './train_events.csv'
    # DescriptiveStat(file, 'csv')
    print('\n---------- Finished reading train_events.csv ----------')

    # Load train_series.parquet
    # file = './train_series.parquet'
    # ReadParquet(file)

    '''
    Coding for determining the time interval
    '''
    file = './train_events_replacement.csv'
    df = pd.read_csv(file)
    gp = df.groupby('series_id')['step'].count()
    gp = pd.DataFrame({'sid': gp.index, 'step_num': gp.values})
    gp['empt_night'] = ''

    for sid in gp.sid:
        df_temp = df[(df.series_id == sid)]
        idx = gp[gp.sid == sid].index[0]
        nights = []

        # Check if each night has a pair of steps
        empty_night = df_temp[df_temp['step'].isna()]['night']
        empty_night = empty_night.unique()
        gp.at[idx, 'empt_night'] = empty_night.tolist()
    
        # Coding for the number of consecutive days that an accelerometer collected records
        max_night = df_temp.groupby('series_id')['night'].max()[0]
        gp.at[idx, 'max_night'] = max_night
        mt_night = gp[gp.sid == sid]['empt_night'].values[0]

        if bool(mt_night) == True:
            for i in range(len(mt_night)):
                if mt_night[i] != 1:
                    if i == 0:
                        con_night = mt_night[i] - 1
                    elif i+1 == len(mt_night) and mt_night[i] < max_night:
                        con_night = max_night - mt_night[i]
                    else:
                        con_night = mt_night[i] - mt_night[i-1] - 1
                elif len(mt_night) == 1:
                    con_night = max_night - mt_night[i]

                nights.append(con_night)

            gp.at[idx, 'max_cont_night'] = max(nights)
        else:
            gp.at[idx, 'max_cont_night'] = max_night

        

    # Configure the dtype of numeric data
    gp['step_num'] = gp['step_num'].astype(np.int8)
    gp['max_night'] = gp['max_night'].astype(np.int8)
    gp['max_cont_night'] = gp['max_cont_night'].astype(np.int8)

    # d = dt.show(gp)
    # print(d._main_url)
    # gp.to_csv('./trE_cont_nights.csv')


    '''
    Checking missing values
    '''
    df_check = pd.read_csv('./train_events.csv')
    for sid in gp.sid:
        max_night = gp[(gp.sid == sid)].max_night.values[0]
        all_nights = df_check[df_check.series_id == sid]['night'].unique()

        for i in range(1, max_night+1):
            if i not in all_nights:
                print(f'Missing entry identified : \tsid [{sid}] \tnight [{i}]')
        

    pass