"""
Module pour les visualisations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def show3d_plot(dataframe, x_col, y_col, z_col, label_col='attack_type', 
                title='Visualisation 3D', figsize=(12, 8)):
    """
    Visualise le dataset en 3D avec différentes couleurs pour chaque classe
    
    Args:
        dataframe: DataFrame avec les données
        x_col: Nom de la colonne pour l'axe X
        y_col: Nom de la colonne pour l'axe Y
        z_col: Nom de la colonne pour l'axe Z
        label_col: Nom de la colonne de labels
        title: Titre du graphique
        figsize: Taille de la figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Obtenir les classes uniques
    unique_labels = dataframe[label_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Tracer chaque classe
    for i, label in enumerate(unique_labels):
        mask = dataframe[label_col] == label
        ax.scatter(
            dataframe[mask][x_col],
            dataframe[mask][y_col],
            dataframe[mask][z_col],
            c=[colors[i]],
            label=label,
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def show3d_attacks_only(dataframe, x_col, y_col, z_col, label_col='attack_type',
                       benign_label='Benign', title='Visualisation 3D - Attaques uniquement',
                       figsize=(12, 8)):
    """
    Visualise uniquement les attaques en 3D
    
    Args:
        dataframe: DataFrame avec les données
        x_col: Nom de la colonne pour l'axe X
        y_col: Nom de la colonne pour l'axe Y
        z_col: Nom de la colonne pour l'axe Z
        label_col: Nom de la colonne de labels
        benign_label: Label pour le trafic bénin
        title: Titre du graphique
        figsize: Taille de la figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Filtrer seulement les attaques
    attacks_df = dataframe[dataframe[label_col] != benign_label]
    
    # Obtenir les types d'attaques uniques
    unique_attacks = attacks_df[label_col].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_attacks)))
    
    # Tracer chaque type d'attaque
    for i, attack_type in enumerate(unique_attacks):
        mask = attacks_df[label_col] == attack_type
        ax.scatter(
            attacks_df[mask][x_col],
            attacks_df[mask][y_col],
            attacks_df[mask][z_col],
            c=[colors[i]],
            label=attack_type,
            alpha=0.7,
            s=30
        )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def show3d_outliers_only(dataframe, x_col, y_col, z_col, outlier_col='outliers',
                         title='Visualisation 3D - Outliers uniquement', figsize=(12, 8)):
    """
    Visualise uniquement les outliers détectés en 3D
    
    Args:
        dataframe: DataFrame avec les données
        x_col: Nom de la colonne pour l'axe X
        y_col: Nom de la colonne pour l'axe Y
        z_col: Nom de la colonne pour l'axe Z
        outlier_col: Nom de la colonne avec les prédictions d'outliers (-1 pour outlier)
        title: Titre du graphique
        figsize: Taille de la figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Filtrer seulement les outliers
    outliers_df = dataframe[dataframe[outlier_col] == -1]
    
    ax.scatter(
        outliers_df[x_col],
        outliers_df[y_col],
        outliers_df[z_col],
        c='red',
        label='Outliers',
        alpha=0.8,
        s=50
    )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_class_distribution(dataframe, label_col='attack_type', figsize=(10, 6)):
    """
    Affiche la distribution des classes
    
    Args:
        dataframe: DataFrame avec les données
        label_col: Nom de la colonne de labels
        figsize: Taille de la figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = dataframe[label_col].value_counts().sort_index()
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts)))
    
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.7)
    ax.set_xlabel('Type d\'attaque')
    ax.set_ylabel('Nombre d\'occurrences')
    ax.set_title('Distribution des classes dans le dataset')
    ax.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6), title='Confusion Matrix'):
    """
    Affiche une matrice de confusion
    
    Args:
        cm: Matrice de confusion
        class_names: Noms des classes
        figsize: Taille de la figure
        title: Titre du graphique
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vraie valeur')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(metrics_df, figsize=(12, 6)):
    """
    Compare les métriques de plusieurs modèles
    
    Args:
        metrics_df: DataFrame avec les métriques (une ligne par modèle)
        figsize: Taille de la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Métriques principales
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Balanced Accuracy', 'MCC']
    available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
    
    x = np.arange(len(metrics_df))
    width = 0.8 / len(available_metrics)
    
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics)/2) * width + width/2
        axes[0].bar(x + offset, metrics_df[metric], width, label=metric)
    
    axes[0].set_xlabel('Modèles')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Comparaison des métriques principales')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # AUPRC et ROC-AUC si disponibles
    auc_metrics = ['AUPRC', 'ROC-AUC']
    available_auc = [m for m in auc_metrics if m in metrics_df.columns]
    
    if available_auc:
        x_auc = np.arange(len(metrics_df))
        width_auc = 0.8 / len(available_auc)
        
        for i, metric in enumerate(available_auc):
            offset = (i - len(available_auc)/2) * width_auc + width_auc/2
            axes[1].bar(x_auc + offset, metrics_df[metric], width_auc, label=metric)
        
        axes[1].set_xlabel('Modèles')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Comparaison AUPRC et ROC-AUC')
        axes[1].set_xticks(x_auc)
        axes[1].set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, top_n=20, figsize=(10, 8)):
    """
    Affiche l'importance des features
    
    Args:
        feature_names: Liste des noms de features
        importances: Array d'importances
        top_n: Nombre de features à afficher
        figsize: Taille de la figure
    """
    # Créer un DataFrame pour faciliter le tri
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Prendre les top N
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} des features les plus importantes')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

