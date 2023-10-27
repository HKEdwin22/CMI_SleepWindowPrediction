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

def PrtAllCols(x, n_col):
   '''
   Set the configure of the environment for Polars
   x : content to be printed
   n_col : number of columns to be displayed in the table or df.width
   '''
   with pl.Config() as cfg:
      cfg.set_tbl_cols(n_col)
      print(x)

def LoadParquet(f):
   '''
   Load the parquet time series data and return the Polars dataframe
   f = file name
   '''

   dfPl = pl.scan_parquet(f).with_columns(
      (
         (pl.col('step').cast(pl.UInt32)),
         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z")),
         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.year().alias("year").cast(pl.UInt16)),
         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.month().alias("month").cast(pl.UInt8)),
         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.day().alias("day").cast(pl.UInt8)),
         (pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.hour().alias("hour").cast(pl.UInt8)),
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z").dt.minute().alias("minute").cast(pl.UInt8)),
         (pl.col('anglez').round(0).cast(pl.Int8)),
         ((pl.col('enmo')*1e3).cast(pl.UInt16))
      )
   ).collect()

   PrtAllCols(dfPl.describe(), dfPl.width)

   return dfPl

# Main program
if __name__ == '__main__':
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   usrAns = False
   if usrAns:
      CheckId()

   if usrAns:
      LoadParquet('./train_series.parquet')
   
   df = pd.read_csv('./trE_cont_nights.csv')
   df = df[df.max_cont_night >= 7]
   for row in range(len(df)):
      daysWanted = {} # key as the start date, item as the end date
      maxNight = df.iloc[row].max_cont_night
      mtNights = df.iloc[row].empt_night
      mtNights = mtNights.strip('][').split(', ')
      
      for i in range(len(mtNights)-1):
         if mtNights[i+1] - mtNights[i] >= maxNight:
            daysWanted[mtNights[i]] = mtNights[i+1]
      df[row].nights_wanted = daysWanted

   # data_transforms = [
      # pl.col('series_id').cast(pl.UInt32)
   #    pl.col('anglez').round(0).cast(pl.Int8)
   #    (pl.col('enmo')*1000).cast(pl.UInt16)
   #    ]
   
   

   

   pass