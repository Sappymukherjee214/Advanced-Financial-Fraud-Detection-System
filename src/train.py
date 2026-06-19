import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

from src.preprocessing import FraudFeatureEngineer, save_preprocessor
from src.models import (
    evaluate_metrics,
    balance_data,
    train_lr,
    train_rf,
    train_xgb,
    train_lgb,
    train_cat,
    train_mlp,
    train_autoencoder,
    run_optuna_tuning
)

def run_pipeline():
    data_dir = "d:/Advanced Financial Fraud Detection System/data"
    model_dir = "d:/Advanced Financial Fraud Detection System/models"
    os.makedirs(model_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train_sampled.csv")
    val_path = os.path.join(data_dir, "val_sampled.csv")
    
    print("Loading sampled datasets...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # Separate features and target
    X_train_raw = df_train.drop(columns=['isFraud'])
    y_train = df_train['isFraud']
    X_val_raw = df_val.drop(columns=['isFraud'])
    y_val = df_val['isFraud']
    
    print("Fitting preprocessing & feature engineering pipeline...")
    # Initialize feature engineer with top 80 features selected by LightGBM gain
    preprocessor = FraudFeatureEngineer(top_k_features=80)
    preprocessor.fit(X_train_raw, y_train)
    
    # Transform train and validation
    print("Transforming datasets...")
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    
    # Save preprocessor
    save_preprocessor(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
    
    metrics_log = {}
    
    # 1. Train supervised baseline models
    print("\n--- Training Supervised Baselines ---")
    
    print("Training Logistic Regression...")
    _, lr_metrics, lr_probs = train_lr(X_train, y_train, X_val, y_val)
    metrics_log['Logistic Regression'] = {k: v for k, v in lr_metrics.items() if k != 'conf_matrix'}
    print(f"LR ROC-AUC: {lr_metrics['roc_auc']:.4f}, PR-AUC: {lr_metrics['pr_auc']:.4f}")
    
    print("Training Random Forest...")
    _, rf_metrics, rf_probs = train_rf(X_train, y_train, X_val, y_val)
    metrics_log['Random Forest'] = {k: v for k, v in rf_metrics.items() if k != 'conf_matrix'}
    print(f"RF ROC-AUC: {rf_metrics['roc_auc']:.4f}, PR-AUC: {rf_metrics['pr_auc']:.4f}")
    
    print("Training XGBoost...")
    # Default scale_pos_weight based on class ratio
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    _, xgb_metrics, xgb_probs = train_xgb(X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos)
    metrics_log['XGBoost'] = {k: v for k, v in xgb_metrics.items() if k != 'conf_matrix'}
    print(f"XGB ROC-AUC: {xgb_metrics['roc_auc']:.4f}, PR-AUC: {xgb_metrics['pr_auc']:.4f}")
    
    print("Training LightGBM...")
    _, lgb_metrics, lgb_probs = train_lgb(X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos)
    metrics_log['LightGBM'] = {k: v for k, v in lgb_metrics.items() if k != 'conf_matrix'}
    print(f"LGB ROC-AUC: {lgb_metrics['roc_auc']:.4f}, PR-AUC: {lgb_metrics['pr_auc']:.4f}")
    
    print("Training CatBoost...")
    _, cat_metrics, cat_probs = train_cat(X_train, y_train, X_val, y_val, scale_pos_weight=scale_pos)
    metrics_log['CatBoost'] = {k: v for k, v in cat_metrics.items() if k != 'conf_matrix'}
    print(f"CatBoost ROC-AUC: {cat_metrics['roc_auc']:.4f}, PR-AUC: {cat_metrics['pr_auc']:.4f}")
    
    # 2. Compare Imbalanced Data Handling Techniques (using LightGBM)
    print("\n--- Comparing Resampling Techniques on LightGBM ---")
    
    # Standard LightGBM already runs with scale_pos_weight. Let's compare with SMOTE, ADASYN, ROS, and RUS.
    print("Applying SMOTE...")
    X_train_smote, y_train_smote = balance_data(X_train, y_train, technique='smote')
    lgb_smote = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_smote.fit(X_train_smote, y_train_smote)
    smote_probs = lgb_smote.predict_proba(X_val)[:, 1]
    smote_metrics = evaluate_metrics(y_val, lgb_smote.predict(X_val), smote_probs)
    print(f"LGB + SMOTE ROC-AUC: {smote_metrics['roc_auc']:.4f}, PR-AUC: {smote_metrics['pr_auc']:.4f}")
    
    print("Applying ADASYN...")
    X_train_adasyn, y_train_adasyn = balance_data(X_train, y_train, technique='adasyn')
    lgb_adasyn = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_adasyn.fit(X_train_adasyn, y_train_adasyn)
    adasyn_probs = lgb_adasyn.predict_proba(X_val)[:, 1]
    adasyn_metrics = evaluate_metrics(y_val, lgb_adasyn.predict(X_val), adasyn_probs)
    print(f"LGB + ADASYN ROC-AUC: {adasyn_metrics['roc_auc']:.4f}, PR-AUC: {adasyn_metrics['pr_auc']:.4f}")
    
    print("Applying Random Oversampling...")
    X_train_ros, y_train_ros = balance_data(X_train, y_train, technique='oversample')
    lgb_ros = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_ros.fit(X_train_ros, y_train_ros)
    ros_probs = lgb_ros.predict_proba(X_val)[:, 1]
    ros_metrics = evaluate_metrics(y_val, lgb_ros.predict(X_val), ros_probs)
    print(f"LGB + Random Oversampling ROC-AUC: {ros_metrics['roc_auc']:.4f}, PR-AUC: {ros_metrics['pr_auc']:.4f}")
    
    print("Applying Undersampling...")
    X_train_under, y_train_under = balance_data(X_train, y_train, technique='undersample')
    lgb_under = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=-1)
    lgb_under.fit(X_train_under, y_train_under)
    under_probs = lgb_under.predict_proba(X_val)[:, 1]
    under_metrics = evaluate_metrics(y_val, lgb_under.predict(X_val), under_probs)
    print(f"LGB + Undersampling ROC-AUC: {under_metrics['roc_auc']:.4f}, PR-AUC: {under_metrics['pr_auc']:.4f}")
    
    # 3. Hyperparameter Tuning using Optuna
    print("\n--- Tuning LightGBM with Optuna ---")
    best_params = run_optuna_tuning(X_train, y_train, X_val, y_val, model_type='lightgbm', n_trials=8)
    print("Best LightGBM parameters found:")
    print(best_params)
    
    # Train final LightGBM model with best parameters
    print("Fitting final optimized LightGBM model...")
    best_lgb = lgb.LGBMClassifier(**best_params, random_state=42)
    best_lgb.fit(X_train, y_train)
    opt_probs = best_lgb.predict_proba(X_val)[:, 1]
    opt_preds = best_lgb.predict(X_val)
    opt_metrics = evaluate_metrics(y_val, opt_preds, opt_probs)
    
    # Overwrite LightGBM metrics log with tuned performance
    metrics_log['LightGBM (Tuned)'] = {k: v for k, v in opt_metrics.items() if k != 'conf_matrix'}
    print(f"Tuned LGB ROC-AUC: {opt_metrics['roc_auc']:.4f}, PR-AUC: {opt_metrics['pr_auc']:.4f}")
    
    # Save best model
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_lgb, f)
    print(f"Saved optimized model to {best_model_path}")
    
    # 4. Deep Learning Classifier (PyTorch MLP)
    print("\n--- Training PyTorch Deep Learning Classifier ---")
    mlp_model, mlp_preds, mlp_probs = train_mlp(X_train, y_train, X_val, y_val, epochs=15, batch_size=256)
    mlp_metrics = evaluate_metrics(y_val, mlp_preds, mlp_probs)
    metrics_log['PyTorch MLP'] = {k: v for k, v in mlp_metrics.items() if k != 'conf_matrix'}
    print(f"MLP ROC-AUC: {mlp_metrics['roc_auc']:.4f}, PR-AUC: {mlp_metrics['pr_auc']:.4f}")
    
    # Save MLP model
    torch.save(mlp_model.state_dict(), os.path.join(model_dir, "mlp_model.pt"))
    
    # 5. Unsupervised Anomaly Detection
    print("\n--- Running Unsupervised Anomaly Detection ---")
    
    print("Training Isolation Forest...")
    # Fits on train set features
    if_model = IsolationForest(contamination=0.035, random_state=42, n_jobs=-1)
    if_model.fit(X_train)
    # Output is -1 for anomaly, 1 for normal
    if_preds = (if_model.predict(X_val) == -1).astype(int)
    # Score anomalies: higher score means less anomalous, so take negative of decision function
    if_scores = -if_model.decision_function(X_val)
    # Scale to [0, 1]
    if_scores_scaled = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())
    if_metrics = evaluate_metrics(y_val, if_preds, if_scores_scaled)
    metrics_log['Isolation Forest'] = {k: v for k, v in if_metrics.items() if k != 'conf_matrix'}
    print(f"IForest ROC-AUC: {if_metrics['roc_auc']:.4f}, PR-AUC: {if_metrics['pr_auc']:.4f}")
    
    print("Training PyTorch Autoencoder Anomaly Detector...")
    ae_model, ae_preds, ae_probs = train_autoencoder(X_train, y_train, X_val, epochs=12, batch_size=256)
    ae_metrics = evaluate_metrics(y_val, ae_preds, ae_probs)
    metrics_log['Autoencoder'] = {k: v for k, v in ae_metrics.items() if k != 'conf_matrix'}
    print(f"Autoencoder ROC-AUC: {ae_metrics['roc_auc']:.4f}, PR-AUC: {ae_metrics['pr_auc']:.4f}")
    
    # Save Autoencoder model
    torch.save(ae_model.state_dict(), os.path.join(model_dir, "autoencoder_model.pt"))
    
    # Save all metrics log
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print(f"\nSaved all metrics comparison to {metrics_path}")
    print("Training Pipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()
