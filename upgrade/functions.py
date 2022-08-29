import pandas as pd
import numpy as np


def get_df_uniques(df):
    attFeatures = []
    for col in df.columns:
        attFeatures.append(
            [col, df[col].dtype, df[col].nunique(), df[col].drop_duplicates().values]
        )
    return pd.DataFrame(attFeatures, columns=[
        'Features', 'Dtype', 'Uniques count', 'Values'
    ])


def get_corr_pairs_thresh(df, size, thresh):
    s = df.corr().abs().unstack().sort_values(ascending=False)
    s = s[s.values < 1]
    for i in range(size * 2):
        if s[i] > thresh:
            if i % 2 == 0:
                print("{:.5f} {}".format(s[i], s.index[i]))


def manage_major_values(df, thresh, drop=False):
    s = pd.Series(dtype="float64")
    for col in df.columns:
        # Ajout dans la Series du pourcentage d'occurence
        # par rapport à la taille de la colonne pour chaque colonnes
        s.loc[col] = df[col].value_counts().iloc[0] / len(df[col]) * 100
    # Récupération des x colonnes avec le pourcentage le plus haut
    # return(s[s > thresh].sort_values(ascending=False))
    return df.drop(columns=s[s > thresh].index.tolist()) if drop else df


def import_data(file, index_col=0):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, index_col=index_col)
    df = reduce_mem_usage(df)
    return df

def reduce_mem_usage(df):
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


def compute_isna_percent(df):
    nas = df.isna().sum()
    return nas[nas > 0].apply(lambda x: round(x / df.shape[0] * 100, 2)).sort_values(ascending=False)
