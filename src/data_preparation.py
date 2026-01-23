"""
Module pour la préparation des données - basé sur le TP de référence
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def get_one_hot_encoded_dataframe(dataframe):
    """
    Retrieves the one hot encoded dataframe

    :param dataframe: input dataframe
    :return: the associated one hot encoded dataframe
    """
    if dataframe is None:
        return None
    return pd.get_dummies(dataframe)


def remove_nan_through_mean_imputation(dataframe):
    """
    Remove NaN (Not a Number) entries through mean imputation using sklearn SimpleImputer

    :param dataframe: input dataframe
    :return: the dataframe with NaN (Not a Number) entries replaced using mean imputation
    """
    if dataframe is None:
        return None
    
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = dataframe.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        df_numeric = dataframe[numeric_cols].copy()
        
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        
        max_float64 = np.finfo(np.float64).max
        min_float64 = np.finfo(np.float64).min
        df_numeric = df_numeric.clip(lower=min_float64 * 0.9, upper=max_float64 * 0.9)
        
        imputer = SimpleImputer(strategy='mean')
        imputed_numeric = imputer.fit_transform(df_numeric)
        imputed_df = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=dataframe.index)
    else:
        imputed_df = pd.DataFrame(index=dataframe.index)
    
    if non_numeric_cols:
        imputed_df = pd.concat([imputed_df, dataframe[non_numeric_cols]], axis=1)
    
    return imputed_df


def remove_infinite_values(dataframe):
    """
    Replace infinite values with NaN, then impute with mean
    
    :param dataframe: input dataframe
    :return: dataframe with infinite values handled
    """
    if dataframe is None:
        return None
    
    df = dataframe.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        df_numeric = df[numeric_cols].copy()
        
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        
        max_float64 = np.finfo(np.float64).max * 0.9
        min_float64 = np.finfo(np.float64).min * 0.9
        df_numeric = df_numeric.clip(lower=min_float64, upper=max_float64)
        
        imputer = SimpleImputer(strategy='mean')
        imputed_numeric = imputer.fit_transform(df_numeric)
        df_numeric = pd.DataFrame(imputed_numeric, columns=numeric_cols, index=df.index)
        
        if non_numeric_cols:
            df = pd.concat([df_numeric, df[non_numeric_cols]], axis=1)
        else:
            df = df_numeric
    else:
        pass
    
    return df


def prepare_features_for_ml(dataframe, label_columns=None, drop_columns=None):
    """
    Prépare les features pour le machine learning:
    - Supprime les colonnes non pertinentes
    - Encode les colonnes catégorielles
    - Impute les valeurs manquantes
    
    :param dataframe: input dataframe
    :param label_columns: list of column names to exclude from features (e.g., labels)
    :param drop_columns: list of column names to drop (e.g., IDs, IPs)
    :return: prepared feature dataframe
    """
    if dataframe is None:
        return None
    
    df = dataframe.copy()
    
    default_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    if drop_columns:
        default_drop.extend(drop_columns)
    
    columns_to_drop = [col for col in default_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    labels = None
    if label_columns:
        labels = df[label_columns].copy()
        df = df.drop(columns=label_columns)
    
    df_encoded = get_one_hot_encoded_dataframe(df)
    
    df_encoded = remove_nan_through_mean_imputation(df_encoded)
    
    df_encoded = remove_infinite_values(df_encoded)
    
    if labels is not None:
        for col in label_columns:
            df_encoded[col] = labels[col].values
    
    return df_encoded

