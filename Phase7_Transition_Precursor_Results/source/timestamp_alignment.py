"""Align by unique target timestamps, rejecting missing requested rows."""
import pandas as pd
def align(df,times):
    df=df.copy();df.target_timestamp=pd.to_datetime(df.target_timestamp)
    assert not df.target_timestamp.duplicated().any()
    df=df.set_index('target_timestamp').loc[pd.to_datetime(times)].reset_index()
    assert len(df)==len(times)
    return df
