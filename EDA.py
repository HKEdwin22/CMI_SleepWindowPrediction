'''
This script is used for EDA of the project.
'''

# Import libraries
import os
import polars as pl
import pandas as pd
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
    
    d = dt.show(df)
    print(d._main_url)
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
    DescriptiveStat(file, 'csv')
    print('\n---------- Finished reading train_events.csv ----------')

    # Load train_series.parquet
    # file = './train_series.parquet'
    # ReadParquet(file)
    
    
    pass