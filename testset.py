# Import libraries
import EDA_sleeplog as es
import polars as pl
import pandas as pd

import time as time

_ = es.ChangeDir()
df = pl.scan_parquet('./sidOfInterest.parquet').collect().to_pandas()
sid = df.series_id.unique()

for s in sid:
    



pass