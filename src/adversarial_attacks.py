"""
Module pour les attaques adversaires contre les modèles de classification (Bonus 20%)
Version corrigée pour Scikit-Learn (Black-Box / Numerical Gradient)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def compute_numerical_gradient(model, X, y, delta=1e-2):
    """
    Calcule une approximation du gradient par différences finies.
    Permet d'attaquer des modèles "boîte noire" (RF, XGBoost, SVM) qui n'ont pas de gradient natif.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    grad_sign = np.zeros_like(X)
    
    # On récupère les probabilités de base
    try:
        probs_base = model.predict_proba(X)
        p_true_base = probs_base[np.arange(n_samples), y]
    except AttributeError:
        return np.sign(np.random.randn(*X.shape))

    features_to_test = range(min(n_features, 50))  # Limiter le nombre de features testées pour la vitesse
    
    for i in features_to_test:
        X_plus = X.copy()
        X_plus[:, i] += delta
        
        probs_plus = model.predict_proba(X_plus)
        p_true_plus = probs_plus[np.arange(n_samples), y]
        
        grad_sign[:, i] = np.where(p_true_plus < p_true_base, 1.0, -1.0)
        
    return grad_sign

def fgsm_attack(model, X, y, epsilon=0.1, scale=True, scaler=None):
    """
    Implémente l'attaque FGSM (Fast Gradient Sign Method) version Black-Box
    """
    X_input = X.copy()
    if isinstance(X_input, pd.DataFrame):
        X_input = X_input.values
    
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_input)
        else:
            X_scaled = scaler.transform(X_input)
    else:
        X_scaled = X_input

    # Calcul du gradient approximatif
    # (On utilise predict_proba pour estimer la direction de l'attaque)
    perturbation = epsilon * compute_numerical_gradient(model, X_scaled, y)
    
    X_adv_scaled = X_scaled + perturbation
    
    if scale and scaler is not None:
        X_adv = scaler.inverse_transform(X_adv_scaled)
    else:
        X_adv = X_adv_scaled
    
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(X_adv, columns=X.columns, index=X.index)
    return X_adv

def pgd_attack(model, X, y, epsilon=0.1, alpha=0.02, num_iter=5, scale=True, scaler=None):
    """
    Implémente l'attaque PGD (Projected Gradient Descent) version Black-Box
    """
    X_input = X.copy()
    if isinstance(X_input, pd.DataFrame):
        X_input = X_input.values
        
    if scale:
        if scaler is None:
            scaler = StandardScaler()
            X_curr = scaler.fit_transform(X_input)
            scaler_used = scaler
        else:
            X_curr = scaler.transform(X_input)
            scaler_used = scaler
    else:
        X_curr = X_input
        scaler_used = None
        
    X_orig = X_curr.copy()

    for _ in range(num_iter):
        grad_sign = compute_numerical_gradient(model, X_curr, y)
        
        X_curr = X_curr + alpha * grad_sign
        
        perturbation = np.clip(X_curr - X_orig, -epsilon, epsilon)
        X_curr = X_orig + perturbation
        
    if scale and scaler_used is not None:
        X_adv = scaler_used.inverse_transform(X_curr)
    else:
        X_adv = X_curr
        
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(X_adv, columns=X.columns, index=X.index)
    return X_adv

def evaluate_adversarial_attack(model, X_original, X_adversarial, y_true, scale=True, scaler=None):
    """
    Évalue l'efficacité de l'attaque
    """

    if isinstance(X_original, pd.DataFrame):
        X_orig_val = X_original.values
        X_adv_val = X_adversarial.values
    else:
        X_orig_val = X_original
        X_adv_val = X_adversarial

    if scale and scaler is not None:

        X_orig_check = scaler.transform(X_orig_val)
        X_adv_check = scaler.transform(X_adv_val)
    else:
        X_orig_check = X_orig_val
        X_adv_check = X_adv_val
    
    y_pred_orig = model.predict(X_orig_check)
    y_pred_adv = model.predict(X_adv_check)
    
    acc_orig = np.mean(y_pred_orig == y_true)
    acc_adv = np.mean(y_pred_adv == y_true)
    success_rate = np.mean(y_pred_orig != y_pred_adv)
    
    return {
        'accuracy_original': acc_orig,
        'accuracy_adversarial': acc_adv,
        'attack_success_rate': success_rate
    }

def adversarial_training(model, X_train, y_train, attack_func, epsilon, n_samples=2000):
    """
    Entraîne un modèle robuste en injectant des exemples adverses.
    
    Args:
        model: Le modèle à ré-entraîner (ex: RandomForest)
        X_train, y_train: Données d'entraînement
        attack_func: La fonction d'attaque (fgsm_attack ou pgd_attack)
        epsilon: La force de l'attaque
        n_samples: Nombre d'exemples adverses à générer (pour ne pas être trop lent)
    """
    
    if n_samples < len(X_train):
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        if isinstance(X_train, pd.DataFrame):
            X_subset = X_train.iloc[indices]
            y_subset = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        else:
            X_subset = X_train[indices]
            y_subset = y_train[indices]
    else:
        X_subset = X_train
        y_subset = y_train

    print(f"Génération de {len(X_subset)} exemples adversaires...")
    X_adv_train = attack_func(model, X_subset, y_subset, epsilon=epsilon, scale=False)
    
    if isinstance(X_train, pd.DataFrame):
        X_combined = pd.concat([X_train, X_adv_train], axis=0)
        y_combined = np.concatenate([y_train, y_subset])
    else:
        X_combined = np.vstack([X_train, X_adv_train])
        y_combined = np.hstack([y_train, y_subset])
        
    print(f"Taille du nouveau dataset d'entraînement : {len(X_combined)}")
    
    print("Ré-entraînement du modèle sur les données mixtes...")
    model.fit(X_combined, y_combined)
    print("Modèle robuste entraîné.")
    
    return model