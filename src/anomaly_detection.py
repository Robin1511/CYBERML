"""
Module pour la détection d'anomalies non supervisée
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


def prepare_numeric_features(dataframe, label_columns=None):
    """
    Prépare les features numériques pour la détection d'anomalies
    
    Args:
        dataframe: DataFrame d'entrée
        label_columns: Colonnes à exclure (labels)
        
    Returns:
        DataFrame avec seulement les features numériques
    """
    if dataframe is None:
        return None
    
    df = dataframe.copy()
    
    # Supprimer les colonnes de labels si spécifiées
    if label_columns:
        df = df.drop(columns=[col for col in label_columns if col in df.columns])
    
    # Sélectionner seulement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclure les colonnes non pertinentes
    exclude_cols = ['Flow ID']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return df[numeric_cols]


def get_isolation_forest_outliers(dataframe, contamination=0.1, random_state=42, **kwargs):
    """
    Détecte les outliers avec Isolation Forest
    
    Args:
        dataframe: DataFrame avec features numériques
        contamination: Proportion attendue d'outliers (sera limité à 0.5 max)
        random_state: Seed pour la reproductibilité
        **kwargs: Paramètres supplémentaires pour IsolationForest
        
    Returns:
        Array de prédictions (-1 pour outliers, 1 pour inliers)
    """
    if dataframe is None:
        return None
    
    # Préparer les features
    features = prepare_numeric_features(dataframe)
    
    # Gérer les valeurs infinies et NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # IsolationForest limite contamination à (0.0, 0.5]
    # Si contamination > 0.5, on utilise 'auto' ou on limite à 0.5
    if contamination > 0.5:
        print(f"  Attention: contamination={contamination:.4f} > 0.5, utilisation de 'auto' pour IsolationForest")
        contamination_param = 'auto'
    else:
        contamination_param = contamination
    
    default_params = {
        'contamination': contamination_param,
        'random_state': random_state,
        'n_estimators': 100
    }
    default_params.update(kwargs)
    
    isolation_forest = IsolationForest(**default_params)
    predictions = isolation_forest.fit_predict(features)
    
    return predictions


def get_lof_outliers(dataframe, contamination=0.1, **kwargs):
    """
    Détecte les outliers avec Local Outlier Factor
    
    Args:
        dataframe: DataFrame avec features numériques
        contamination: Proportion attendue d'outliers (sera limité à 0.5 max)
        **kwargs: Paramètres supplémentaires pour LOF
        
    Returns:
        Array de prédictions (-1 pour outliers, 1 pour inliers)
    """
    if dataframe is None:
        return None
    
    # Préparer les features
    features = prepare_numeric_features(dataframe)
    
    # Gérer les valeurs infinies et NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # LOF limite contamination à (0.0, 0.5]
    # Si contamination > 0.5, on utilise 'auto' ou on limite à 0.5
    if contamination > 0.5:
        print(f"  Attention: contamination={contamination:.4f} > 0.5, utilisation de 'auto' pour LOF")
        contamination_param = 'auto'
    else:
        contamination_param = contamination
    
    default_params = {
        'contamination': contamination_param,
        'n_neighbors': 20,
        'novelty': False
    }
    default_params.update(kwargs)
    
    lof = LocalOutlierFactor(**default_params)
    predictions = lof.fit_predict(features)
    
    return predictions


def get_elliptic_envelope_outliers(dataframe, contamination=0.1, random_state=42, **kwargs):
    """
    Détecte les outliers avec Elliptic Envelope (covariance robuste)
    ✅ Alternative RAPIDE à One-Class SVM (10-100x plus rapide)
    
    Args:
        dataframe: DataFrame avec features numériques
        contamination: Proportion attendue d'outliers (sera limité à 0.5 max)
        random_state: Seed pour la reproductibilité
        **kwargs: Paramètres supplémentaires pour EllipticEnvelope
        
    Returns:
        Array de prédictions (-1 pour outliers, 1 pour inliers)
    """
    if dataframe is None:
        return None
    
    # Préparer les features
    features = prepare_numeric_features(dataframe)
    
    # Gérer les valeurs infinies et NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.mean())
    
    # EllipticEnvelope limite contamination à (0.0, 0.5]
    if contamination > 0.5:
        print(f"  Attention: contamination={contamination:.4f} > 0.5, limitation à 0.5 pour EllipticEnvelope")
        contamination_param = 0.5
    else:
        contamination_param = contamination
    
    default_params = {
        'contamination': contamination_param,
        'random_state': random_state,
        'support_fraction': None  # Utiliser tous les points pour estimer la covariance
    }
    default_params.update(kwargs)
    
    elliptic_envelope = EllipticEnvelope(**default_params)
    predictions = elliptic_envelope.fit_predict(features)
    
    return predictions


def evaluate_anomaly_detection(y_true_labels, predictions, attack_label=1):
    """
    Évalue la performance de la détection d'anomalies
    
    Args:
        y_true_labels: Vraies labels (0 pour normal, 1 pour attaque)
        predictions: Prédictions (-1 pour outlier, 1 pour inlier)
        attack_label: Label utilisé pour les attaques dans y_true_labels
        
    Returns:
        Dictionnaire avec les métriques
    """
    # Convertir les prédictions: -1 -> 1 (outlier/attaque), 1 -> 0 (inlier/normal)
    y_pred = (predictions == -1).astype(int)
    
    # Convertir les vraies labels si nécessaire
    if isinstance(y_true_labels, pd.Series):
        y_true = y_true_labels.values
    else:
        y_true = y_true_labels
    
    # Calculer les métriques
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }
    
    return metrics
