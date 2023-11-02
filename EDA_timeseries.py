# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
import time

import matplotlib.pyplot as plt
import seaborn as sns

# Function or Class
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

def ExtractTimeSeries():
   '''
   Extract data of interest from .parquet file (56min32s)
   '''
   x1 = pd.read_csv('./sleepLog_stepWanted.csv')
   y = pl.scan_parquet('./train_series.parquet').with_columns(
         (pl.col('step').cast(pl.UInt32)),
         (pl.col('anglez').round(0).cast(pl.Int8)),
         ((pl.col('enmo')*1e3).cast(pl.UInt16))
      ).clone().clear().collect()

   for sid in x1.sid:
   
      start = datetime.strptime(x1[x1.sid == sid]['start'].values[0], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
      end = datetime.strptime(x1[x1.sid == sid]['end'].values[0], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
      
      lf = pl.scan_parquet('./train_series.parquet').filter(
         (pl.col('series_id') == sid) &
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") >= start) & 
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") <= end)
         ).with_columns(
            (pl.col('step').cast(pl.UInt32)),
            (pl.col('anglez').round(0).cast(pl.Int8)),
            ((pl.col('enmo')*1e3).cast(pl.UInt16))
         )

      y.vstack(lf.collect(), in_place=True)
   
   # y.write_parquet('./ExtractTimeSeries.parquet')


# Main program
if __name__ == '__main__':
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   usrAns = False
   if usrAns:
      CheckId()
      ExtractTimeSeries()
   
   # lf = LoadParquet('./train_series.parquet', None)

   startExe = time.time()

   start = '2018-08-28T20:37:00-0400'
   start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
   end = '2018-08-29T08:37:00-0400'
   end = datetime.strptime(end, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
   
   lf = pl.scan_parquet('./ExtractedTimeSeries.parquet').filter(
      (pl.col('series_id') == '038441c925bb') &
      (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") >= start) &
      (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") <= end)
      ).select(['series_id', 'timestamp', 'anglez'])
   
   endExe = time.time()
   print(f'Execution time : {(endExe-startExe):.2f} seconds')

   # plt.figure(figsize=(12,9))
   # sns.lineplot(data=lf.collect(), x='timestamp', y='anglez')
   # plt.tight_layout(h_pad=5, w_pad=5)
   # plt.show()

   import plotly.graph_objs as go
   fig = go.Figure(data=go.Scatter(x=lf.collect()['timestamp'], 
                           y=lf.collect()['anglez'],
                           marker_color='indianred', text="counts"))
   fig.update_layout({"title": 'Anglez of an accelerometer within 2 hours',
                     "xaxis": {"title":"Time"},
                     "yaxis": {"title":"Angle"},
                     "showlegend": False})
   # fig.write_image("by-month.png",format="png", width=1000, height=600, scale=3)
   fig.show()

   pass