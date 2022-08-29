"""Utility functions for data exploration and cleaning."""
from typing import Any, Union
import pandas as pd
import numpy as np


def get_df_uniques(df: pd.DataFrame) -> pd.DataFrame:
    att_features = []
    for col in df.columns:
        att_features.append(
            [col, df[col].dtype, df[col].nunique(), df[col].drop_duplicates().values]
        )
    return pd.DataFrame(att_features, columns=[
        'Features', 'Dtype', 'Uniques count', 'Values'
    ])


def get_corr_pairs_thresh(df: pd.DataFrame, size: int, thresh: float):
    s = df.corr().abs().unstack().sort_values(ascending=False)
    s = s[s.values < 1]
    for i in range(size * 2):
        if s[i] > thresh and i % 2 == 0:
            print("{:.5f} {}".format(s[i], s.index[i]))


def manage_major_values(df: pd.DataFrame, thresh: float, drop: bool = False) -> pd.DataFrame:
    s = pd.Series(dtype="float64")
    for col in df.columns:
        # add to series percent representation of major value for each column
        s.loc[col] = df[col].value_counts().iloc[0] / len(df[col]) * 100
    # get n columns with the highest major value
    return df.drop(columns=s[s > thresh].index.tolist()) if drop else df


def import_data(csv_file: str, index_col: Union[int, Any] = 0) -> pd.DataFrame:
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(csv_file, parse_dates=True, keep_date_col=True, index_col=index_col)
    return reduce_mem_usage(df)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.3f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def compute_isna_percent(df: pd.DataFrame) -> pd.Series:
    nas = df.isna().sum()
    return nas[nas > 0].apply(lambda x: round(x / df.shape[0] * 100, 2)).sort_values(ascending=False)
