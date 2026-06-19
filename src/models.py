import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = 0.5
        metrics['pr_auc'] = 0.0
    metrics['conf_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

# -------------------------------------------------------------
# 1. PyTorch Models: Deep Learning & Anomaly Detection
# -------------------------------------------------------------

class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super(FraudMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, epochs=20, batch_size=256, lr=0.001):
    X_tr_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    
    # Simple DataLoader
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = FraudMLP(X_train.shape[1]).to(device)
    # Class weights to handle imbalance in cross entropy
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    # Inference
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
        val_preds = (val_probs >= 0.5).astype(int)
        
    return model, val_preds, val_probs


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_autoencoder(X_train, y_train, X_val, epochs=15, batch_size=256, lr=0.001):
    # Train autoencoder ONLY on non-fraud (normal) cases
    normal_idx = (y_train == 0)
    X_train_normal = X_train[normal_idx]
    
    X_tr_t = torch.tensor(X_train_normal.values, dtype=torch.float32)
    dataset = TensorDataset(X_tr_t, X_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Autoencoder(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            
    # Compute reconstruction error on val set
    model.eval()
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_val = model(X_val_t)
        # Compute MSE row-wise
        errors = torch.mean((reconstructed_val - X_val_t) ** 2, dim=1).cpu().numpy()
        
    # Categorize as fraud if reconstruction error is high (e.g. 95th percentile)
    threshold = np.percentile(errors, 95)
    val_preds = (errors > threshold).astype(int)
    # Scale reconstruction error to [0, 1] as "anomaly probability"
    max_err = errors.max() if errors.max() > 0 else 1
    val_probs = errors / max_err
    
    return model, val_preds, val_probs

# -------------------------------------------------------------
# 2. Resampling & Balancing Techniques
# -------------------------------------------------------------

def balance_data(X_train, y_train, technique='smote'):
    if technique == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    elif technique == 'adasyn':
        try:
            adasyn = ADASYN(random_state=42)
            X_res, y_res = adasyn.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"ADASYN failed, falling back to SMOTE: {e}")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
    elif technique == 'oversample':
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)
    elif technique == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train
    return X_res, y_res

# -------------------------------------------------------------
# 3. Supervised Classifiers & Hyperparameter Tuning
# -------------------------------------------------------------

def train_lr(X_train, y_train, X_val, y_val, class_weight='balanced'):
    model = LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    return model, evaluate_metrics(y_val, preds, probs), probs

def train_rf(X_train, y_train, X_val, y_val, class_weight='balanced'):
    model = RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    return model, evaluate_metrics(y_val, preds, probs), probs

def train_xgb(X_train, y_train, X_val, y_val, scale_pos_weight=1.0):
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    return model, evaluate_metrics(y_val, preds, probs), probs

def train_lgb(X_train, y_train, X_val, y_val, scale_pos_weight=1.0):
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    return model, evaluate_metrics(y_val, preds, probs), probs

def train_cat(X_train, y_train, X_val, y_val, scale_pos_weight=1.0):
    model = cb.CatBoostClassifier(
        iterations=150,
        learning_rate=0.1,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        random_seed=42,
        thread_count=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    return model, evaluate_metrics(y_val, preds, probs), probs

# -------------------------------------------------------------
# 4. Optuna Hyperparameter Optimization
# -------------------------------------------------------------

def run_optuna_tuning(X_train, y_train, X_val, y_val, model_type='lightgbm', n_trials=10):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        if model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0),
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0),
                'random_state': 42,
                'n_jobs': -1
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 250),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 30.0),
                'random_seed': 42,
                'thread_count': -1,
                'verbose': 0
            }
            model = cb.CatBoostClassifier(**params)
            
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        
        # Optimize PR-AUC (Average Precision)
        score = average_precision_score(y_val, probs)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[{model_type.upper()}] Best PR-AUC: {study.best_value:.4f}")
    return study.best_params
