# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd


# Main program
if __name__ == '__main__':
    
   # Change directory to load the data
   myDir = es.ChangeDir()
   
   # Load data
   data_transforms = [
      pl.col('anglez').round(0).cast(pl.Int8), # Casting anglez to 8 bit integer
      (pl.col('enmo')*1000).cast(pl.UInt16), # Convert enmo to 16 bit uint
      ]
   
   # df = pl.scan_parquet('./train_series.parquet').with_columns(data_transforms).collect()
   trId = pl.scan_parquet('./train_series.parquet').select('series_id').unique()
   sid = pd.read_csv('./trE_cont_nights.csv')['sid']

   

   pass