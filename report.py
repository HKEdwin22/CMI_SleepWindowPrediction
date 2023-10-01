# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change directory to access the raw data
dir_old = os.getcwd()    
os.chdir('../../')      # work under "My Documents"
dir_MyDoc = os.getcwd()
target_path = 'DSAI\Kaggle_Competitions\CMI_Detect Sleep States\RawData'
os.chdir(os.path.join(dir_MyDoc, target_path))

# Plot for target distribution
df = pd.read_csv('./train_events.csv')
percentage = [50, 50]

ax = sns.countplot(df, x='event', palette='PuBuGn_r')
patches = ax.patches
for i in range(len(patches)):
   x = patches[i].get_x() + patches[i].get_width()/2
   y = patches[i].get_height()+.05
   ax.annotate('{:.1f}%'.format(percentage[i]), (x, y), ha='center')
plt.show()