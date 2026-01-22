"""
Module pour l'évaluation des modèles avec toutes les métriques requises
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from sklearn.preprocessing import LabelEncoder


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None, average='binary'):
    """
    Calcule toutes les métriques requises pour l'évaluation
    
    Args:
        y_true: Vraies labels (peut être textuel ou numérique)
        y_pred: Prédictions (peut être textuel ou numérique)
        y_pred_proba: Probabilités prédites (pour AUPRC et ROC-AUC)
        average: Type de moyenne pour les métriques multi-classes
        
    Returns:
        Dictionnaire avec toutes les métriques
    """
    # Convertir en numpy arrays si nécessaire
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Convertir en arrays numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Vérifier les types et encoder si nécessaire
    y_true_encoded = y_true.copy()
    y_pred_encoded = y_pred.copy()
    
    if len(y_true) > 0 and len(y_pred) > 0:
        # Vérifier si les éléments sont des strings
        y_true_is_str = isinstance(y_true[0], str) or (hasattr(y_true, 'dtype') and y_true.dtype == 'object')
        y_pred_is_str = isinstance(y_pred[0], str) or (hasattr(y_pred, 'dtype') and y_pred.dtype == 'object')
        
        # Vérifier si les éléments sont numériques
        y_true_is_num = isinstance(y_true[0], (int, np.integer, float, np.floating))
        y_pred_is_num = isinstance(y_pred[0], (int, np.integer, float, np.floating))
        
        # Si les deux sont textuels, encoder les deux
        if y_true_is_str and y_pred_is_str:
            le = LabelEncoder()
            # Fit sur la combinaison des deux pour avoir tous les labels
            all_labels = np.concatenate([y_true, y_pred])
            le.fit(all_labels)
            y_true_encoded = le.transform(y_true)
            y_pred_encoded = le.transform(y_pred)
        
        # Si y_true est textuel mais y_pred est numérique
        elif y_true_is_str and y_pred_is_num:
            le = LabelEncoder()
            le.fit(y_true)
            y_true_encoded = le.transform(y_true)
            # Convertir y_pred en entiers et vérifier la correspondance
            y_pred_int = y_pred.astype(np.int32)
            # S'assurer que les valeurs sont dans la plage valide
            unique_true_encoded = np.unique(y_true_encoded)
            min_true = np.min(unique_true_encoded)
            max_true = np.max(unique_true_encoded)
            # Ajuster y_pred pour qu'il soit dans la même plage
            y_pred_min = np.min(y_pred_int)
            if y_pred_min != min_true:
                y_pred_encoded = y_pred_int - y_pred_min + min_true
            else:
                y_pred_encoded = y_pred_int
            # Clipper pour être sûr
            y_pred_encoded = np.clip(y_pred_encoded, min_true, max_true)
        
        # Si y_pred est textuel mais y_true est numérique
        elif y_pred_is_str and y_true_is_num:
            le = LabelEncoder()
            le.fit(y_pred)
            y_pred_encoded = le.transform(y_pred)
            y_true_encoded = y_true.astype(np.int32)
        
        # Si les deux sont numériques
        else:
            y_true_encoded = y_true.astype(np.int32)
            y_pred_encoded = y_pred.astype(np.int32)
    
    metrics = {}
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    # Precision
    metrics['precision'] = precision_score(y_true_encoded, y_pred_encoded, average=average, zero_division=0)
    
    # Recall
    metrics['recall'] = recall_score(y_true_encoded, y_pred_encoded, average=average, zero_division=0)
    
    # F1-score
    metrics['f1_score'] = f1_score(y_true_encoded, y_pred_encoded, average=average, zero_division=0)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
    
    # Balanced accuracy
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true_encoded, y_pred_encoded)
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true_encoded, y_pred_encoded)
    
    # AUPRC (Area Under Precision-Recall Curve)
    if y_pred_proba is not None:
        try:
            if average == 'binary':
                metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
            else:
                # Pour multi-classes, calculer la moyenne macro
                metrics['auprc'] = average_precision_score(
                    y_true, y_pred_proba, average='macro'
                )
        except Exception as e:
            metrics['auprc'] = None
            print(f"Warning: Impossible de calculer AUPRC: {e}")
    else:
        metrics['auprc'] = None
    
    # ROC-AUC (bonus)
    if y_pred_proba is not None:
        try:
            if average == 'binary':
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Pour multi-classes, utiliser 'ovr' ou 'ovo'
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba, average='macro', multi_class='ovr'
                )
        except Exception as e:
            metrics['roc_auc'] = None
            print(f"Warning: Impossible de calculer ROC-AUC: {e}")
    else:
        metrics['roc_auc'] = None
    
    return metrics


def print_metrics_summary(metrics, model_name="Model"):
    """
    Affiche un résumé des métriques
    
    Args:
        metrics: Dictionnaire de métriques
        model_name: Nom du modèle
    """
    print(f"\n{'='*60}")
    print(f"Métriques pour {model_name}")
    print(f"{'='*60}")
    
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    print(f"\nMétriques principales:")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
    print(f"  MCC:           {metrics['mcc']:.4f}")
    
    if metrics['auprc'] is not None:
        print(f"  AUPRC:         {metrics['auprc']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:       {metrics['roc_auc']:.4f}")
    
    print(f"{'='*60}\n")


def compare_models(metrics_dict):
    """
    Compare plusieurs modèles et retourne un DataFrame comparatif
    
    Args:
        metrics_dict: Dictionnaire {nom_modèle: métriques}
        
    Returns:
        DataFrame comparatif
    """
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {
            'Model': model_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'Accuracy': metrics['accuracy'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'MCC': metrics['mcc'],
        }
        
        if metrics['auprc'] is not None:
            row['AUPRC'] = metrics['auprc']
        if metrics['roc_auc'] is not None:
            row['ROC-AUC'] = metrics['roc_auc']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Génère un rapport de classification détaillé
    
    Args:
        y_true: Vraies labels
        y_pred: Prédictions
        target_names: Noms des classes
        
    Returns:
        Rapport de classification
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

