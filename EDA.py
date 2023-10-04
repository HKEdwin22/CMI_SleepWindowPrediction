'''
This script is used for EDA of the project.
'''

# Import libraries
import os
import re
import polars as pl
import pandas as pd
import numpy as np
import dtale

import seaborn as sns
import matplotlib.pyplot as plt


def ChangeDir():
    '''
    Change directory to ./RawData at the beginning and return the directory of "My Documents"
    '''
    dir_old = os.getcwd()    
    os.chdir('../../')      # work under "My Documents"
    dir_MyDoc = os.getcwd()
    target_path = 'DSAI\Kaggle_Competitions\CMI_Detect Sleep States\RawData'
    os.chdir(os.path.join(dir_MyDoc, target_path))

    return dir_MyDoc

def DescriptiveStat(x, t):
    '''
    Generate an overall statistical description of the dataset and return the dataframe
    x : input dataset
    t : file type (csv or parquet)
    '''

    if t == 'csv':
        df = pd.read_csv(x)

    print(df.info())
    print(df.nunique())
    print(df.describe())

    return df

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

class IdentifyContradictions:

    def __init__(self, x) -> None:
        '''
        Handle contradictory data on target variable
        x : input dataset (raw data)
        '''
        self.x = x
        pass
    
    # Determining the time interval 1 - preparing the dataset
    def contradictions(self):
        '''
        Identify the accelerometers with suspicious records. Replace corresponding steps and timestamps with np.nan and save a new .csv file. Return the new dataframe "gp"
        '''
        # Identify
        gp = self.x.groupby('series_id')['step'].count() 
        print(gp[gp % 2 == 1])

        target_replace = gp[gp % 2 == 1]
        df_contradict = pd.DataFrame({'series_id':target_replace.index, 
                                'night':[20, 30, 10, 7, 17], 
                                'additional event':['onset', 'wakeup', 'wakeup', 'wakeup', 'wakeup']
                                })
        print(df_contradict)

        # Replace
        for i in df_contradict.index:
            sid = df_contradict['series_id'][i]
            night = df_contradict['night'][i]
            event = df_contradict['additional event'][i]

            self.x.loc[(self.x['series_id'] == sid) & (self.x['night'] == night) & (self.x['event'] == event), 'step'] = np.nan
            self.x.loc[(self.x['series_id'] == sid) & (self.x['night'] == night) & (self.x['event'] == event), 'timestamp'] = np.nan
        
        # Save the updated dataset into .csv
        self.x.to_csv('./train_events_replacement.csv')

        self.replacedDF = gp

    def nightsNoRecord(self):
        '''
        Identify nights without records and return the relevant dataframe
        '''
        gp = pd.DataFrame({'sid': self.replacedDF['index'], 'step_num': self.replacedDF['values']})
        gp['empt_night'] = ''
        print(gp.head())

        for sid in gp.sid:
            df_temp = self.x[(self.x.series_id == sid)]
            idx = gp[gp.sid == sid].index[0]
            nights = []

            # Check if each night has a pair of steps
            empty_night = df_temp[df_temp['step'].isna()]['night']
            empty_night = empty_night.unique()
            gp.at[idx, 'empt_night'] = empty_night.tolist()

            # Count the number of consecutive days that an accelerometer collected records
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

        gp.to_csv('./trE_cont_nights.csv')

        self.replacedDF = gp

def ExtractDateTime(x):
    '''
    Extract the year, month, day, hour and minute
    x: input dataset
    '''
    df_temp = pd.read_csv(x, index_col=0)
    df_temp = df_temp.dropna()
    df_temp['UTC_timestamp'] = pd.to_datetime(df_temp['timestamp'], utc=True)

    df_temp['year'] = df_temp['UTC_timestamp'].dt.year
    df_temp['month'] = df_temp['UTC_timestamp'].dt.month
    df_temp['day'] = df_temp['UTC_timestamp'].dt.day
    df_temp['hour'] = df_temp['UTC_timestamp'].dt.hour
    df_temp['minute'] = df_temp['UTC_timestamp'].dt.minute
    df_temp = df_temp.drop('timestamp', axis=1)
    df_temp = df_temp.drop('UTC_timestamp', axis=1)

    return df_temp

def CheckMissingVal(x):
        '''
        Check missing entries of each accelerometer. Return the input dataframe, the trE_cont_nights dataframe and the result as a dictionary
        x : input dataset (.csv)
        '''
        df_check = pd.read_csv(x)
        meter_msval = {}
        gp = pd.read_csv('./trE_cont_nights.csv', index_col=0)
        
        for sid in gp.sid:
            max_night = gp[gp[gp.sid == sid]].max_night.values[0]
            all_nights = df_check[df_check(df_check.series_id == sid)]['night'].unique()

            for i in range(1, max_night+1):
                if i not in all_nights:
                    print(f'Missing entry identified : \tsid [{sid}] \tnight [{i}]')
                    if meter_msval.has_key(sid):
                        meter_msval[sid].append(i)
                    else:
                        meter_msval[sid] = [i]

        return df_check, gp, meter_msval


# Main program
if __name__ == '__main__':

    dir_mydoc = ChangeDir()
    
    # Overview of train_events.csv
    file = './train_events.csv'
    df = DescriptiveStat(file, 'csv')
    print(df.nunique())
    print('\n---------- Finished reading train_events.csv ----------')

    # Looking into the details of the data
    d = dtale.show(df)
    print(d._main_url)

    # Load train_series.parquet
    # file = './train_series.parquet'
    # ReadParquet(file)

    '''
    Determining the time interval
    '''
    # Identify the contradictory steps and timestamps
    Int = IdentifyContradictions(df)
    Int.contradictions()
    Int.nightsNoRecord()
    dfNoContra = Int.replacedDF

    # Looking into the dataset
    dfNoContra['max_cont_night'].describe()
    percent = pd.DataFrame({'Day': [dfNoContra['max_cont_night'].quantile(0.35), dfNoContra['max_cont_night'].quantile(0.43)], 
                            'Percentage' : [1-0.35, 1-0.43],
                            'Number of Samples': [(277*.65), 277*.57]
                            })
    print(percent)

    '''
    Exploiting the dataset - creating a new table
    '''
    dfUTC = ExtractDateTime('./train_events_replacement.csv')

    # Compute the number of steps for each night and store in a new dataframe
    col_sid = []
    col_night = []
    col_diff = []

    for sid in gp['sid'].values:
        max_night = gp['max_night'].values

        for night in range(1, max_night+1):
            step_on = df_temp[(df_temp['sid' == sid]) & (df_temp['night' == night]) & (df_temp['event' == 'onset'])]['step'].values[0]
            step_up = df_temp[(df_temp['sid' == sid]) & (df_temp['night' == night]) & (df_temp['event' == 'wakeup'])]['step'].values[0]
            diff = step_up - step_on
            
            col_sid.append(sid)
            col_night.append(night)
            col_diff.append(diff)
        
    df_diff = pd.DataFrame({'sid': col_sid,
                            'night': col_night,
                            'step_number': col_diff
                            })

    # Check missing values
    dfRaw, dfContN, missVal = CheckMissingVal('./train_events.csv')   

    
    pass