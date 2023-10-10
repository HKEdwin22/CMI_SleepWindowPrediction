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

from scipy import stats, spatial
from sklearn import preprocessing

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
        print(self.replacedDF.index)
        gp = pd.DataFrame({'sid': self.replacedDF.index, 'step_num': self.replacedDF.values})
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
            max_night = gp[gp.sid == sid].max_night.values[0]
            all_nights = df_check[df_check.series_id == sid]['night'].unique()

            for i in range(1, max_night+1):
                if i not in all_nights:
                    print(f'Missing entry identified : \tsid [{sid}] \tnight [{i}]')
                    if sid in meter_msval.keys():
                        meter_msval[sid].append(i)
                    else:
                        meter_msval[sid] = [i]

        return df_check, gp, meter_msval

def CorrectRawData(x):
    '''
    Add the missing nights to the raw dataset for better analysis
    x : raw dataset
    '''
    dfNew = pd.DataFrame({'series_id': ['137771d19ca2', '1f96b9668bdf', 'c7b1283bb7eb', 'c7d693f24684', 'e11b9d69f856'], 
                          'night': [1, 1, 1, 1, 1], 'event': ['onset', 'onset', 'onset', 'onset', 'onset'], 
                          'step': [np.nan, np.nan, np.nan, np.nan, np.nan], 
                          'timestamp': [np.nan, np.nan, np.nan, np.nan, np.nan]
                          })
    x = pd.concat([x, dfNew], ignore_index=True)
    dfNew = pd.DataFrame({'series_id': ['137771d19ca2', '1f96b9668bdf', 'c7b1283bb7eb', 'c7d693f24684', 'e11b9d69f856'], 
                          'night': [1, 1, 1, 1, 1], 'event': ['wakeup', 'wakeup', 'wakeup', 'wakeup', 'wakeup'], 
                          'step': [np.nan, np.nan, np.nan, np.nan, np.nan], 
                          'timestamp': [np.nan, np.nan, np.nan, np.nan, np.nan]
                          })
    x = pd.concat([x, dfNew], ignore_index=True)

    return x

def ComputeStepSleep(x, t, df, mV):
    '''
    Compute number of steps during a sleep window and the sleep duration
    x : dfNoContra
    t : dfUTC
    df : (corrected) raw dataset
    mV : missVal
    '''        
    dfDummy = pd.DataFrame({'sid': [], 'night': [], 'step_number': [], 'sleep_duration': []})

    for sid in x['sid'].values:
        df_diff = {'sid': [], 'night': [], 'step_number': [], 'sleep_duration': []}
        max_night = x[x.sid == sid]['max_night'].values[0]
        mtNight = x[x.sid == sid].empt_night.values[0] # This returns the list as stored in the dataframe
        # map(int, re.findall(r'\d+', mtNight))
        if (mV != []) & (sid in mV):
            mtNight = mV[sid] + mtNight

        for night in range(1, max_night+1):
            if night not in mtNight:
                valOn = t[(t['series_id'] == sid) & (t['night'] == night) & (t['event'] == 'onset')]['step'].values[0]
                valUp = t[(t['series_id'] == sid) & (t['night'] == night) & (t['event'] == 'wakeup')]['step'].values[0]
                diff = valUp - valOn

                df_diff['sid'].append(sid)
                df_diff['night'].append(night)
                df_diff['step_number'].append(diff)

                valOn = pd.to_datetime(df[(df.series_id == sid) & (df.night == night) & (df.event == 'onset')]['timestamp'], utc=True)
                valUp = pd.to_datetime(df[(df.series_id == sid) & (df.night == night) & (df.event == 'wakeup')]['timestamp'], utc=True)
                diff = pd.Series(valUp.values - valOn.values, name='duration')
        
                df_diff['sleep_duration'].append(diff)
    
        temp = pd.DataFrame(df_diff)
        dfDummy = pd.concat([dfDummy, temp], ignore_index=True)
    
    dfDummy = dfDummy.astype({'night': 'int8'})
    dfDummy = dfDummy.astype({'step_number': 'int16'})
    # dfDummy.to_csv('differences.csv', index=True)

    return dfDummy

def DecomposeTimeDelta(x):
        '''
        Decompose the sleep duration into day, hour, minute etc.
        x: input dataset (dfDiff)
        '''
        X = x.sleep_duration
        days, hours, mins = [], [], []

        for t in X:
            days.append(t[0].components.days)
            hours.append(t[0].components.hours)
            mins.append(t[0].components.minutes)

        x['slp_days'] = days
        x['slp_hrs'] = hours
        x['slp_mins'] = mins            
        x['total'] = x.slp_hrs*60 + x.slp_mins
        x.to_csv('./differences.csv', index=True)

        return x 

def LookIntoDetail(x):
    '''
    Run dtale and look into the data
    x: input dataset
    '''
    d = dtale.show(x)
    print(d._main_url)

# Main program
if __name__ == '__main__':

    dir_mydoc = ChangeDir()
    
    # Overview of train_events.csv
    file = './train_events.csv'
    df = DescriptiveStat(file, 'csv')

    usrAns = False
    if usrAns:
        df = CorrectRawData(df)
        print(df.nunique())
    print('\n---------- Finished reading train_events.csv ----------')

    # Looking into the details of the raw data
    usrAns = False
    LookIntoDetail if usrAns else print('---------- Skip the detail of the raw dataset ----------')

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
    print('---------- Dataset has no more contradiction ----------')

    # Looking into the dataset
    dfNoContra['max_cont_night'].describe()
    percent = pd.DataFrame({'Day': [dfNoContra['max_cont_night'].quantile(.3156), dfNoContra['max_cont_night'].quantile(0.4149)], 
                            'Percentage' : [1-0.3156, 1-0.4129],
                            'Number of Samples': [277*(1-0.3156), 277*(1-0.4129)]
                            })
    print(percent)

    '''
    Exploiting the dataset
    '''
    dfUTC = ExtractDateTime('./train_events_replacement.csv')

    # Check missing values
    usrAns = False
    if usrAns:
        dfRaw, dfContN, missVal = CheckMissingVal('./train_events.csv')   
        print('---------- Missing nights checked ----------\n')
    else:
        print('---------- Missing nights passed ----------\n')
        missVal = []

    # Compute the number of steps for each night and store in a new dataframe
    usrAns = False
    if usrAns:
        dfDiff = ComputeStepSleep(dfNoContra, dfUTC, df, missVal)
        DecomposeTimeDelta(dfDiff)
        print('---------- Number of steps and sleep duration computed ----------\n')
    else:
        print('---------- Skip computing the number of steps and sleep duration ----------\n')

    df = pd.read_csv('./differences.csv', index_col=0)

    # Visualise the relation between the number of steps and total sleep duration
    usrAns = False
    if usrAns:
        plt.figure(figsize=(16,9))
        sns.scatterplot(x='total', y='step_number', data=df)
        plt.xlabel('Total Sleep Duration (mins)')
        plt.ylabel('Number of Steps (in a sleep window per night)')

        plt.tight_layout()
        plt.show()

    # Pearson correlation coefficient of the number of steps and total sleep duration
    result = stats.pearsonr(x=df.total, y=df.step_number)
    print(f'Pearson correlation coefficient:\t{result[0]}\t\t\tp-value:\t{result[1]}')

    # Determine if the correlation coefficient is statistically significant
    # def BivariateNormalCheck(x):
    '''
    Check if the variables have a bivariate normal distribution
    x: input dataframe
    '''
    # Extract useful data
    x1 = df['total'].to_numpy()
    x1 = x1.reshape((len(x1),1))
    x2 = df['step_number'].to_numpy()
    x2 = x2.reshape((len(x2),1))
    x = np.stack((x1, x2), axis=1).T 
    x = x.reshape((2,len(x1)))

    # Compute the inverse covariance matrix and the mean
    covInv = np.linalg.pinv(np.cov(x))
    mu = [x[0].mean(), x[1].mean()]
    x = x.T

    # Compute Mahalanobis distance
    mahaDis = []
    for i in x:
        mahaDis.append(spatial.distance.mahalanobis(i.tolist(), mu, VI=covInv))
    
    # Generate a Chi-square distribution
    np.random.seed(7)
    chi2 = np.random.chisquare(df=len(df)-1, size=len(df))
    chi2Q = [np.quantile(chi2, i/len(df)) for i in range(0, len(df))]
    
    # Plot the Chi-square QQ Plot
    plt.figure(figsize=(12,9))
    sns.scatterplot(x=mahaDis, y=chi2Q)
    plt.show()
   

    # stpStd = preprocessing(df.step_number)
    # totalStd = preprocessing(df.total)
    # result = stats.pearsonr(stpStd, totalStd)
    # print(result)
    # stats.ttest_ind(stpStd, totalStd, equal_var=True)

    pass