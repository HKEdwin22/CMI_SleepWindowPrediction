# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm


import plotly.graph_objs as go

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

def BuildDateSet():
   '''
   Build dataset for training or testing
   '''
   x1 = pl.scan_csv('./sleepLog_stepWanted.csv').select(['sid', 'start']).collect().to_pandas()
   x2 = pd.read_csv('./train_events_replacement.csv', index_col=None)

   y = pl.scan_parquet('./train_series.parquet').with_columns(
         (pl.col('step').cast(pl.UInt32)),
         (pl.col('anglez').round(0).cast(pl.Int8))
      ).select(['series_id', 'step', 'timestamp', 'anglez']).clone().clear().collect()
   y = y.with_columns(pl.lit('').alias('state'))

   for sid in tqdm(x1.sid, desc='Dataset under construction'):
         
      # Extract the onset/wakeup time from csv files
      refNight = x1[(x1['sid'] == sid)]['start'].values[0]
      wantedNight = x2[(x2['series_id'] == sid) & (x2['timestamp'] == refNight)]['night'].values[0] + 1
      OnsetTime = x2[(x2['series_id'] == sid) & (x2['night'] == wantedNight) & (x2['event'] == 'onset')]['timestamp'].values[0]
      WakeupTime = x2[(x2['series_id'] == sid) & (x2['night'] == wantedNight) & (x2['event'] == 'wakeup')]['timestamp'].values[0]

      startOnset = datetime.strptime(OnsetTime, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) - timedelta(hours=2)
      endOnset = datetime.strptime(OnsetTime, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) + timedelta(hours=2)
      startWakeup = datetime.strptime(WakeupTime, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) - timedelta(hours=2)
      endWakeup = datetime.strptime(WakeupTime, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) + timedelta(hours=2)
      
      # Extract the time series for onset time
      lf = pl.scan_parquet('./train_series.parquet').filter(
         (pl.col('series_id') == sid) &
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") >= startOnset) & 
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") <= endOnset)
         ).with_columns(
            (pl.col('step').cast(pl.UInt32)),
            (pl.col('anglez').round(0).cast(pl.Int8)),
            (pl.lit('awake').alias('state'))
         ).select(['series_id', 'step', 'timestamp', 'anglez', 'state'])

      y.vstack(lf.collect(), in_place=True)

      # Extract the time series for wakeup time
      lf = pl.scan_parquet('./train_series.parquet').filter(
         (pl.col('series_id') == sid) &
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") >= startWakeup) & 
         (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") <= endWakeup)
         ).with_columns(
            (pl.col('step').cast(pl.UInt32)),
            (pl.col('anglez').round(0).cast(pl.Int8)),
            (pl.lit('sleep').alias('state'))
         ).select(['series_id', 'step', 'timestamp', 'anglez', 'state'])

      y.vstack(lf.collect(), in_place=True)

      y.write_parquet('./ExtactTimeSeries_2hrs_labeled.parquet')

def PrintTimeSeriesSample():

   x0 = '2018-12-26T19:58:00-0500'
   start = datetime.strptime(x0, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) - timedelta(hours=8, minutes=0)
   x1 = '2018-12-27T01:37:00-0500'
   end = datetime.strptime(x1, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) + timedelta(hours=8, minutes=0)
   
   lf = pl.scan_parquet('./ExtractedTimeSeries.parquet').filter(
      (pl.col('series_id') == '0402a003dae9') &
      (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") >= start) &
      (pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%Z") <= end)
      ).select(['series_id', 'timestamp', 'anglez'])

   fig = go.Figure(data=go.Scatter(x=lf.collect()['timestamp'], 
                           y=lf.collect()['anglez'],
                           marker_color='blue', text="counts"))
   fig.update_layout({"title": 'Enmo of an accelerometer from onset to wakeup (+/- 2hrs)',
                     "xaxis": {"title":"Time"},
                     "yaxis": {"title":"Angle"},
                     "showlegend": False})
   fig.add_vline(x=x0, line_width=3, line_dash='dash', line_color='red')
   fig.add_vline(x=x1, line_width=3, line_dash='dash', line_color='red')
   fig.add_vrect(x0=x0, x1=x1, line_width=0, fillcolor="red", opacity=0.2)
   # fig.write_image("by-month.png",format="png", width=1000, height=600, scale=3)
   fig.show()


# def RollingAvg(x):
#    '''
#    Build a new table holding the rolling average of anglez
#    x : input file
#    '''
#    accelerometer = pl.scan_parquet(x).select('series_id').collect().to_pandas()
#    for sid in tqdm(accelerometer.series_id):
#       targetSeries = pl.scan_parquet(x).filter(
#          (pl.col('series_id') == sid) &
#          (pl.col())
#       )

# Main program
if __name__ == '__main__':

   startExe = time.time()
    
   # Change directory to load the data
   myDir = es.ChangeDir()

   usrAns = False
   if usrAns:
      CheckId()
      ExtractTimeSeries()
      PrintTimeSeriesSample()
      
   # lf = LoadParquet('./train_series.parquet', None)
   BuildDateSet()


   
   endExe = time.time()
   print(f'Execution time : {(endExe-startExe):.2f} seconds')
   pass