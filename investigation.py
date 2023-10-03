# Import libraries
import os
import pandas as pd

# Change directory to access the raw data
dir_old = os.getcwd()    
os.chdir('../../')      # work under "My Documents"
dir_MyDoc = os.getcwd()
target_path = 'DSAI\Kaggle_Competitions\CMI_Detect Sleep States\RawData'
os.chdir(os.path.join(dir_MyDoc, target_path))

# Load the dataset
# Extract the year, month, day, hour and minute
gp = pd.read_csv('./trE_cont_nights.csv')
df_temp = pd.read_csv('./train_events_replacement.csv', index_col=0)
df_temp = df_temp.dropna()
df_temp['UTC_timestamp'] = pd.to_datetime(df_temp['timestamp'], utc=True)

df_temp['year'] = df_temp['UTC_timestamp'].dt.year
df_temp['month'] = df_temp['UTC_timestamp'].dt.month
df_temp['day'] = df_temp['UTC_timestamp'].dt.day
df_temp['hour'] = df_temp['UTC_timestamp'].dt.hour
df_temp['minute'] = df_temp['UTC_timestamp'].dt.minute
df_temp = df_temp.drop('timestamp', axis=1)
df_temp = df_temp.drop('UTC_timestamp', axis=1)


# Compute the number of steps for each night and store in a new dataframe
col_sid = []
col_night = []
col_diff = []

for sid in gp['sid']:
    max_night = gp[gp.sid == sid]['max_night'].values[0]
    mt_night = gp[gp.sid == sid]['empt_night'].values[0]

    for night in range(1, max_night+1):
        if night not in mt_night:
            step_on = df_temp[(df_temp.series_id == sid) & (df_temp.night == night) & (df_temp.event == 'onset')]['step'].values[0]
            step_up = df_temp[(df_temp.series_id == sid) & (df_temp.night == night) & (df_temp.event == 'wakeup')]['step'].values[0]
            diff = step_up - step_on
            
            col_sid.append(sid)
            col_night.append(night)
            col_diff.append(diff)
    
df_diff = pd.DataFrame({'sid': col_sid,
                        'night': col_night,
                        'step_number': col_diff
                        })