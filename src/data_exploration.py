"""
Module pour l'exploration des données - basé sur le TP de référence
"""
import pandas as pd
import numpy as np


def get_column_names(dataframe):
    """
    Get the name of columns in the dataframe

    :param dataframe: input dataframe
    :return: name of columns
    """
    return dataframe.columns.tolist()


def get_nb_of_dimensions(dataframe):
    """
    Retrieves the number of dimensions of a pandas dataframe

    :param dataframe: input dataframe
    :return: number of dimensions
    """
    return dataframe.shape[1]


def get_nb_of_rows(dataframe):
    """
    Get the number of rows

    :param dataframe: input dataframe
    :return: number of rows
    """
    if dataframe is None:
        return None
    return dataframe.shape[0]


def get_number_column_names(dataframe):
    """
    Get the name of numeric columns

    :param dataframe: input dataframe
    :return: name of numeric columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include=['number']).columns.tolist()


def get_object_column_names(dataframe):
    """
    Get the name of object columns

    :param dataframe: input dataframe
    :return: name of object columns
    """
    if dataframe is None:
        return None
    return dataframe.select_dtypes(include=['object']).columns.tolist()


def get_unique_values(dataframe, column_name):
    """
    Get the unique values for a given column

    :param dataframe: input dataframe
    :param column_name: target column label
    :return: unique values for a given column
    """
    if dataframe is None:
        return None
    return dataframe[column_name].unique()


def get_class_distribution(dataframe, label_column='attack_type'):
    """
    Get the distribution of classes in the dataset
    
    :param dataframe: input dataframe
    :param label_column: name of the label column
    :return: Series with class distribution
    """
    if dataframe is None or label_column not in dataframe.columns:
        return None
    return dataframe[label_column].value_counts()


def get_nan_rates(dataframe):
    """
    Calculate the rate of NaN values for each column
    
    :param dataframe: input dataframe
    :return: Series with NaN rates
    """
    if dataframe is None:
        return None
    return (dataframe.isnull().sum() / len(dataframe)) * 100


def get_zero_rates(dataframe, numeric_only=True):
    """
    Calculate the rate of zero values for numeric columns
    
    :param dataframe: input dataframe
    :param numeric_only: if True, only consider numeric columns
    :return: Series with zero rates
    """
    if dataframe is None:
        return None
    
    if numeric_only:
        numeric_cols = get_number_column_names(dataframe)
        if not numeric_cols:
            return pd.Series()
        df_subset = dataframe[numeric_cols]
    else:
        df_subset = dataframe
    
    return ((df_subset == 0).sum() / len(df_subset)) * 100


def get_statistics_by_class(dataframe, feature_column, label_column='attack_type'):
    """
    Get statistics for a feature grouped by class
    
    :param dataframe: input dataframe
    :param feature_column: name of the feature column
    :param label_column: name of the label column
    :return: DataFrame with statistics by class
    """
    if dataframe is None or feature_column not in dataframe.columns or label_column not in dataframe.columns:
        return None
    
    return dataframe.groupby(label_column)[feature_column].describe()

