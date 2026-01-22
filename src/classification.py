"""
Module pour les algorithmes de classification supervisée
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ClassificationModels:
    """
    Classe pour gérer les différents modèles de classification
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}  # Pour encoder les labels textuels en numériques
        self.feature_names = None
        
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        """
        Entraîne un modèle XGBoost
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement (peut être textuel ou numérique)
            X_test: Features de test (optionnel)
            y_test: Labels de test (optionnel)
            **kwargs: Paramètres supplémentaires pour XGBoost
            
        Returns:
            Modèle entraîné
        """
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if y_test is not None and isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Encoder les labels si ce sont des strings
        y_train_encoded = y_train.copy()
        y_test_encoded = y_test.copy() if y_test is not None else None
        
        # Vérifier si les labels sont textuels
        is_textual = (len(y_train) > 0 and isinstance(y_train[0], str)) or \
                     (hasattr(y_train, 'dtype') and y_train.dtype == 'object')
        
        if is_textual:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            self.label_encoders['XGBoost'] = label_encoder
            
            if y_test is not None:
                y_test_encoded = label_encoder.transform(y_test)
        
        # S'assurer que les labels sont des entiers et commencent à 0 (requis par XGBoost)
        y_train_encoded = np.asarray(y_train_encoded, dtype=np.int32)
        if y_test_encoded is not None:
            y_test_encoded = np.asarray(y_test_encoded, dtype=np.int32)
        
        # Vérifier que les labels commencent à 0
        if len(np.unique(y_train_encoded)) > 0:
            min_label = np.min(y_train_encoded)
            if min_label != 0:
                y_train_encoded = y_train_encoded - min_label
                if y_test_encoded is not None:
                    y_test_encoded = y_test_encoded - min_label
        
        # Déterminer si c'est multi-classes ou binaire
        n_classes = len(np.unique(y_train_encoded))
        is_multiclass = n_classes > 2
        
        # Choisir la métrique appropriée
        if is_multiclass:
            eval_metric = 'mlogloss'  # Pour multi-classes
            objective = 'multi:softprob'  # Pour multi-classes avec probabilités
        else:
            eval_metric = 'logloss'  # Pour binaire
            objective = 'binary:logistic'  # Pour binaire
        
        default_params = {
            'random_state': 42,
            'eval_metric': eval_metric,
            'objective': objective,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1
        }
        default_params.update(kwargs)
        
        model = xgb.XGBClassifier(**default_params)
        
        if X_test is not None and y_test_encoded is not None:
            model.fit(X_train, y_train_encoded, 
                     eval_set=[(X_test, y_test_encoded)],
                     verbose=False)
        else:
            model.fit(X_train, y_train_encoded)
        
        self.models['XGBoost'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Entraîne un modèle Random Forest
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            **kwargs: Paramètres supplémentaires pour Random Forest
            
        Returns:
            Modèle entraîné
        """
        default_params = {
            'random_state': 42,
            'n_estimators': 100,
            'max_depth': 20,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['RandomForest'] = model
        return model
    
    def train_svm(self, X_train, y_train, scale=True, use_linear=True, **kwargs):
        """
        Entraîne un modèle SVM
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement (peut être textuel ou numérique)
            scale: Si True, normalise les features
            use_linear: Si True, utilise LinearSVC (plus rapide), sinon SVC avec kernel RBF
            **kwargs: Paramètres supplémentaires pour SVM
            
        Returns:
            Modèle entraîné
        """
        # Encoder les labels si ce sont des strings
        y_train_encoded = y_train.copy()
        
        if y_train.dtype == 'object' or isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            self.label_encoders['SVM'] = label_encoder
        
        if use_linear:
            # Utiliser LinearSVC pour la vitesse (beaucoup plus rapide)
            default_params = {
                'random_state': 42,
                'C': 1.0,
                'max_iter': 1000,
                'dual': False  # dual=False est plus rapide pour n_samples > n_features
            }
            # Filtrer les paramètres non valides pour LinearSVC (kernel, probability, gamma, etc.)
            linear_svc_params = ['C', 'dual', 'fit_intercept', 'intercept_scaling', 
                                'loss', 'max_iter', 'multi_class', 'penalty', 
                                'random_state', 'tol', 'verbose']
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in linear_svc_params}
            default_params.update(filtered_kwargs)
            
            model = LinearSVC(**default_params)
        else:
            # Utiliser SVC avec kernel RBF (plus lent mais plus flexible)
            default_params = {
                'random_state': 42,
                'kernel': 'rbf',
                'probability': True,
                'C': 1.0,
                'gamma': 'scale'
            }
            default_params.update(kwargs)
            
            model = SVC(**default_params)
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['SVM'] = scaler
            model.fit(X_train_scaled, y_train_encoded)
        else:
            model.fit(X_train, y_train_encoded)
        
        self.models['SVM'] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, scale=True, **kwargs):
        """
        Entraîne un modèle de régression logistique
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            scale: Si True, normalise les features
            **kwargs: Paramètres supplémentaires pour Logistic Regression
            
        Returns:
            Modèle entraîné
        """
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'multi_class': 'ovr'
        }
        default_params.update(kwargs)
        
        model = LogisticRegression(**default_params)
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['LogisticRegression'] = scaler
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        self.models['LogisticRegression'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, scale=True, **kwargs):
        """
        Entraîne un réseau de neurones
        
        Args:
            X_train: Features d'entraînement
            y_train: Labels d'entraînement
            scale: Si True, normalise les features
            **kwargs: Paramètres supplémentaires pour MLPClassifier
            
        Returns:
            Modèle entraîné
        """
        default_params = {
            'random_state': 42,
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        default_params.update(kwargs)
        
        model = MLPClassifier(**default_params)
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['NeuralNetwork'] = scaler
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        self.models['NeuralNetwork'] = model
        return model
    
    def predict(self, model_name, X, return_proba=False):
        """
        Fait des prédictions avec un modèle
        
        Args:
            model_name: Nom du modèle
            X: Features
            return_proba: Si True, retourne aussi les probabilités
            
        Returns:
            Prédictions (et probabilités si return_proba=True)
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        model = self.models[model_name]
        
        # Appliquer le scaler si nécessaire
        if model_name in self.scalers:
            X = self.scalers[model_name].transform(X)
        
        y_pred = model.predict(X)
        
        # Décoder les prédictions si un label encoder existe (pour avoir le même format que les vraies labels)
        if model_name in self.label_encoders:
            try:
                # S'assurer que y_pred est un array numpy d'entiers
                y_pred_int = np.asarray(y_pred, dtype=np.int32)
                y_pred = self.label_encoders[model_name].inverse_transform(y_pred_int)
            except Exception as e:
                # Si le décodage échoue, garder les prédictions numériques
                print(f"Warning: Impossible de décoder les prédictions: {e}")
                pass
        
        if return_proba:
            # LinearSVC n'a pas predict_proba, utiliser decision_function
            if isinstance(model, LinearSVC):
                # Utiliser decision_function et convertir en probabilités avec sigmoid
                try:
                    from scipy.special import expit
                    decision_scores = model.decision_function(X)
                    # Convertir en probabilités pour classification binaire
                    if len(decision_scores.shape) == 1:  # Binary classification
                        y_proba = expit(decision_scores)
                    else:  # Multi-class
                        # Pour multi-class, utiliser softmax
                        from scipy.special import softmax
                        y_proba = softmax(decision_scores, axis=1)
                        # Retourner les probabilités de la classe positive (dernière colonne)
                        if y_proba.shape[1] == 2:
                            y_proba = y_proba[:, 1]
                except ImportError:
                    # Si scipy n'est pas disponible, utiliser une approximation simple
                    decision_scores = model.decision_function(X)
                    if len(decision_scores.shape) == 1:
                        # Approximation sigmoid simple
                        y_proba = 1 / (1 + np.exp(-decision_scores))
                    else:
                        # Approximation softmax simple
                        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                        y_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                        if y_proba.shape[1] == 2:
                            y_proba = y_proba[:, 1]
            else:
                y_proba = model.predict_proba(X)
                # Pour la classification binaire, retourner les probabilités de la classe positive
                if y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]
            return y_pred, y_proba
        
        return y_pred
    
    def get_feature_importance(self, model_name):
        """
        Obtient l'importance des features pour un modèle
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Array ou None si le modèle ne supporte pas feature_importances_
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Pour SVM et Logistic Regression, utiliser les coefficients
            return np.abs(model.coef_[0])
        else:
            return None

