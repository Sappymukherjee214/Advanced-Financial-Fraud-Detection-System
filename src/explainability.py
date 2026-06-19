import os
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular

class FraudRiskScorer:
    def __init__(self, threshold_low=20, threshold_medium=50, threshold_high=80):
        self.t_low = threshold_low
        self.t_med = threshold_medium
        self.t_high = threshold_high
        
    def calculate_score(self, prob):
        # Convert probability to a 0-100 score
        return int(round(prob * 100))
        
    def get_category_details(self, score):
        if score <= self.t_low:
            return {
                'level': 'Low Risk',
                'color': '#00ff88', # Light neon green
                'action': 'Auto-Approve',
                'recommendation': 'Transaction exhibits standard behavior. Approve immediately.'
            }
        elif score <= self.t_med:
            return {
                'level': 'Medium Risk',
                'color': '#ffee00', # Neon yellow
                'action': 'Flag & Monitor',
                'recommendation': 'Subtle behavioral variations detected. Monitor client patterns.'
            }
        elif score <= self.t_high:
            return {
                'level': 'High Risk',
                'color': '#ffaa00', # Orange
                'action': 'Require Step-up Authentication',
                'recommendation': 'Trigger Multi-Factor (OTP) authentication before allowing transaction.'
            }
        else:
            return {
                'level': 'Critical Risk',
                'color': '#ff003c', # Bright neon red
                'action': 'Decline & Freeze Account',
                'recommendation': 'High resemblance to known fraud clusters. Reject and lock credentials.'
            }

# -------------------------------------------------------------
# SHAP & LIME Explainer Wrappers
# -------------------------------------------------------------

def setup_lime_explainer(X_train, feature_names, categorical_features=None):
    # Fit LimeTabularExplainer on training data
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['Legit', 'Fraud'],
        mode='classification',
        categorical_features=categorical_features,
        verbose=False,
        random_state=42
    )
    return explainer

def get_lime_explanation_for_instance(explainer, instance, predict_fn):
    # Generate local prediction explanations using LIME
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=8
    )
    # returns list of (feature_rule, weight)
    return exp.as_list()

def get_shap_explanation_for_instance(model, X_background, instance):
    # Tree explainer is very fast for XGBoost/LightGBM
    # Use background dataset for reference (sampled 100 rows)
    explainer = shap.Explainer(model, X_background)
    
    # instance must be a 2D array or row
    if len(instance.shape) == 1:
        instance = instance.reshape(1, -1)
        
    shap_values = explainer(instance)
    return shap_values
