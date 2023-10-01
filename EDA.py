'''
This script is used for EDA of the project.
'''

# Import libraries
import os
import polars as pl
import pandas as pd

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
    else:
        df = pl.read_
    print(df.info() + '\n')
    print(df.nunique() + '\n')
    print(df.describe() + '\n')

# Main program
if __name__ == '__main__':

    dir_old = os.getcwd()    
    os.chdir('../../')      # work under "My Documents"
    dir_MyDoc = os.getcwd()
    target_path = 'DSAI\Kaggle_Competitions\CMI_Detect Sleep States\RawData'
    os.chdir(os.path.join(dir_MyDoc, target_path))
    
    file = './train_events.csv'
    DescriptiveStat(file)
    
    pass