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
file = './train_events.csv'
df = pd.read_csv(file)

# Identify the accelerometers with suspicious records
gp = df.groupby('series_id')['step'].count() 
print(gp[gp % 2 == 1])