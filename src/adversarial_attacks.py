"""
Module pour les attaques adversaires contre les modèles de classification (Bonus 20%)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def fgsm_attack(model, X, y, epsilon=0.01, scale=True, scaler=None):
    """
    Implémente l'attaque FGSM (Fast Gradient Sign Method)
    
    Args:
        model: Modèle de classification entraîné (doit avoir predict_proba)
        X: Features d'entrée
        y: Vraies labels
        epsilon: Magnitude de la perturbation
        scale: Si True, utilise le scaler pour normaliser
        scaler: Scaler pré-entraîné (si None, crée un nouveau)
        
    Returns:
        X_adversarial: Features adversaires
    """
    X = X.copy()
    
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
    else:
        X_scaled = X.copy()
    
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    
    X_scaled = X_scaled.astype(np.float32)
    X_scaled.requires_grad = True
    
    try:
        y_pred_proba = model.predict_proba(X_scaled)
    except:
        y_pred = model.predict(X_scaled)
        y_pred_proba = np.zeros((len(y_pred), 2))
        y_pred_proba[np.arange(len(y_pred)), y_pred] = 1.0
    
    y_one_hot = np.zeros_like(y_pred_proba)
    y_one_hot[np.arange(len(y)), y] = 1.0
    
    loss = -np.sum(y_one_hot * np.log(y_pred_proba + 1e-10), axis=1)
    X_adversarial_scaled = X_scaled.copy()
    for i in range(len(X_scaled)):
        perturbation = epsilon * np.sign(X_scaled[i] - X_scaled[i].mean())
        X_adversarial_scaled[i] = X_scaled[i] + perturbation
    
    if scale and scaler is not None:
        X_adversarial = scaler.inverse_transform(X_adversarial_scaled)
    else:
        X_adversarial = X_adversarial_scaled
    
    return X_adversarial


def pgd_attack(model, X, y, epsilon=0.01, alpha=0.001, num_iter=10, scale=True, scaler=None):
    """
    Implémente l'attaque PGD (Projected Gradient Descent)
    
    Args:
        model: Modèle de classification entraîné
        X: Features d'entrée
        y: Vraies labels
        epsilon: Magnitude maximale de la perturbation
        alpha: Taux d'apprentissage pour chaque itération
        num_iter: Nombre d'itérations
        scale: Si True, utilise le scaler
        scaler: Scaler pré-entraîné
        
    Returns:
        X_adversarial: Features adversaires
    """
    X = X.copy()
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
    else:
        X_scaled = X.copy()
    
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    
    X_adversarial_scaled = X_scaled.copy().astype(np.float32)
    
    for _ in range(num_iter):
        try:
            y_pred_proba = model.predict_proba(X_adversarial_scaled)
        except:
            y_pred = model.predict(X_adversarial_scaled)
            y_pred_proba = np.zeros((len(y_pred), 2))
            y_pred_proba[np.arange(len(y_pred)), y_pred] = 1.0
        perturbation = alpha * np.sign(X_adversarial_scaled - X_scaled)
        X_adversarial_scaled = X_adversarial_scaled + perturbation
        
        delta = X_adversarial_scaled - X_scaled
        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = delta / (delta_norm + 1e-10) * np.minimum(delta_norm, epsilon)
        X_adversarial_scaled = X_scaled + delta
    if scale and scaler is not None:
        X_adversarial = scaler.inverse_transform(X_adversarial_scaled)
    else:
        X_adversarial = X_adversarial_scaled
    
    return X_adversarial


def evaluate_adversarial_attack(model, X_original, X_adversarial, y_true, scale=True, scaler=None):
    """
    Évalue l'impact d'une attaque adverse sur un modèle
    
    Args:
        model: Modèle à évaluer
        X_original: Features originales
        X_adversarial: Features adversaires
        y_true: Vraies labels
        scale: Si True, utilise le scaler
        scaler: Scaler pré-entraîné
        
    Returns:
        Dictionnaire avec les métriques d'évaluation
    """
    if scale and scaler is not None:
        X_orig_scaled = scaler.transform(X_original)
    else:
        X_orig_scaled = X_original
    
    y_pred_original = model.predict(X_orig_scaled)
    accuracy_original = np.mean(y_pred_original == y_true)
    
    if scale and scaler is not None:
        X_adv_scaled = scaler.transform(X_adversarial)
    else:
        X_adv_scaled = X_adversarial
    
    y_pred_adversarial = model.predict(X_adv_scaled)
    accuracy_adversarial = np.mean(y_pred_adversarial == y_true)
    
    attack_success_rate = np.mean(y_pred_original != y_pred_adversarial)
    
    return {
        'accuracy_original': accuracy_original,
        'accuracy_adversarial': accuracy_adversarial,
        'accuracy_drop': accuracy_original - accuracy_adversarial,
        'attack_success_rate': attack_success_rate
    }


def adversarial_training(model_class, X_train, y_train, X_val, y_val, 
                        attack_func=fgsm_attack, epsilon=0.01, 
                        num_epochs=5, **model_kwargs):
    """
    Entraîne un modèle avec adversarial training pour améliorer sa robustesse
    
    Args:
        model_class: Classe du modèle à entraîner
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        attack_func: Fonction d'attaque à utiliser
        epsilon: Magnitude de la perturbation
        num_epochs: Nombre d'époques d'adversarial training
        **model_kwargs: Arguments pour le modèle
        
    Returns:
        Modèle entraîné avec adversarial training
    """
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train)
    
    for epoch in range(num_epochs):
        X_adv = attack_func(model, X_train, y_train, epsilon=epsilon)
        
        X_combined = np.vstack([X_train, X_adv])
        y_combined = np.hstack([y_train, y_train])
        model.fit(X_combined, y_combined)
        val_acc = model.score(X_val, y_val)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_acc:.4f}")
    
    return model

