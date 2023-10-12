# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd
import numpy as np

def CheckId():
   '''
   Check if the accelerometers in train_series.parquet exist in train_events.csv
   '''
   trId = pl.scan_parquet('./train_series.parquet').select('series_id').unique()
   trId = trId.sort('series_id').select('series_id').collect().get_column('series_id')
   sid = pd.read_csv('./trE_cont_nights.csv')['sid'].sort_values()
   idWanted = []

   for i in range(len(sid)):
      if trId[i] == sid[i]:
         idWanted.append(sid[i])

   if len(idWanted) == len(sid):
      print('---------- All accelerometers in the train series are in the train events ----------\n')
   else:
      print('---------- Some accelerometers are not in the train events. Further check is needed ----------\n')


# Main program
if __name__ == '__main__':
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   usrAns = False
   if usrAns:
      CheckId()
   
   # Load data
   # data_transforms = [
   #    pl.col('anglez').round(0).cast(pl.Int8), # Casting anglez to 8 bit integer
   #    (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
   #    ]
   
   

   

   pass