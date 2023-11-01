# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
import time

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

def LoadParquet(f, n=None):
   '''
   Load the parquet time series data and return the Polars dataframe
   f : file name
   n : max rows to be loaded
   '''

   lf = pl.scan_parquet(f, n_rows=n).with_columns(
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
   )
   
   dfPl = lf.collect()

   PrtAllCols(dfPl.describe(), dfPl.width)

   return lf


# Main program
if __name__ == '__main__':
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   usrAns = False
   if usrAns:
      CheckId()
   
   start = time.time()

   lf = LoadParquet('./train_series.parquet', None)
   df = pd.read_csv('./sleepLog_stepWanted.csv')

   dftsSID = lf.select('series_id').unique().collect().to_series().to_list()

   '''
   Extract data of interest from .parquet file
   '''
   def ExtractTimeSeries(x, x1, s):
      '''
      Extract data of interest from .parquet file
      x : time series lazyframe
      x1 : .csv dataframe ('./sleepLog_stepWanted.csv')
      s : series_id of interest
      '''
      newTS = x.clone()
      for sid in s:
         if sid in x1['sid'].values:
            start = datetime.strptime(x1[x1.sid == sid]['start'].values[0], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            end = datetime.strptime(x1[x1.sid == sid]['end'].values[0], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)

            newTS.vstack(x.filter
                           (
                              (pl.col('series_id') == sid) &
                              (pl.col('timestamp') >= start) & 
                              (pl.col('timestamp') <= end)
                           ),
                           in_place=True)
            
         x = x.filter(pl.col('series_id') != sid)
      newTS.write_parquet('./sidOfInterest.parquet')
   
   
   ExtractTimeSeries(lf, df, dftsSID[:3])
      
   # data_transforms = [
      # pl.col('series_id').cast(pl.UInt32)
   #    pl.col('anglez').round(0).cast(pl.Int8)
   #    (pl.col('enmo')*1000).cast(pl.UInt16)
   #    ]
   
   end = time.time()
   print(f'Time of Execution : {end-start}')

   

   pass