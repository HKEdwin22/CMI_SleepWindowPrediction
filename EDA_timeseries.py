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

def LoadParquet(f, n=None):
   '''
   Load the parquet time series data and return the Polars dataframe
   f : file name
   n : max rows to be loaded
   '''

   dfPl = pl.scan_parquet(f, n_rows=n).with_columns(
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

   dfts = LoadParquet('./train_series.parquet', 1000)
   df = pd.read_csv('./sleepLog_stepWanted.csv', index_col=0)
   dfDateTime = pd.read_csv('./trE_replacement_ExtDateTime.csv', index_col=0)
   
   '''
   Extract period of interest for involved accelerometers
   '''
   dftsNew = dfts.clone().clear()
   for sid in dfts['series_id'].unique():
      if sid in df['sid'].values:
         nights = df[df.sid == sid].nights_wanted.values[0]
         if nights == 'all':
            dftsNew.vstack(dfts.filter(dfts['series_id'] == sid))
         else:
            if ',' in nights:
               nights = nights.strip('][').split(', ') # case 1: "['21 to 28', '6 to 19']"
               for n in range(len(nights)):
                  nights[n] = nights[n].strip('\'') # ['21 to 28', '6 to 19']
               for n in nights:
                  nightPair = n.strip('][\'').split(' to ') # case 2: "['15 to 29']" or item of handled case 1
                  dftsNew.vstack(dfts.filter((dfts['series_id'] == sid) & (dfts['step'] >= int(nightPair[0])) & (dfts['step'] <= int(nightPair[1]))), in_place=True)
            else:
               nightPair = nights.strip('][\'').split(' to ') # case 2: "['15 to 29']" or item of handled case 1
               dftsNew.vstack(dfts.filter((dfts['series_id'] == sid) & (dfts['step'] >= int(nightPair[0])) & (dfts['step'] <= int(nightPair[1]))), in_place=True)
   dftsNew.write_parquet('./sidOfInterest.parquet')

      

   # data_transforms = [
      # pl.col('series_id').cast(pl.UInt32)
   #    pl.col('anglez').round(0).cast(pl.Int8)
   #    (pl.col('enmo')*1000).cast(pl.UInt16)
   #    ]
   
   

   

   pass