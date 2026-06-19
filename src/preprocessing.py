import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb

class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, top_k_features=80):
        self.top_k_features = top_k_features
        self.fitted = False
        
        # Aggregation maps fitted on train
        self.card1_amt_mean = {}
        self.card1_amt_std = {}
        self.card1_addr1_amt_mean = {}
        self.card1_addr1_amt_std = {}
        
        # Preprocessing sub-pipelines
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        self.numerical_cols = []
        self.categorical_cols = []
        self.selected_features = []
        self.feature_names_out_ = []
        
    def _create_features(self, df):
        df_out = df.copy()
        
        # Time-based features
        df_out['Transaction_Hour'] = (df_out['TransactionDT'] // 3600) % 24
        df_out['Transaction_Day'] = (df_out['TransactionDT'] // (3600 * 24)) % 7
        
        # Amount-based features
        df_out['TransactionAmt_log'] = np.log1p(df_out['TransactionAmt'])
        df_out['TransactionAmt_decimal'] = df_out['TransactionAmt'] - np.trunc(df_out['TransactionAmt'])
        
        # Email domain matching
        df_out['Email_Match'] = (df_out['P_emaildomain'].astype(str) == df_out['R_emaildomain'].astype(str)).astype(int)
        
        # Missingness indicator count
        df_out['Null_Count'] = df_out.isnull().sum(axis=1)
        
        # Card + address interaction key
        df_out['card1_addr1'] = df_out['card1'].astype(str) + "_" + df_out['addr1'].astype(str)
        
        # Map aggregations
        # card1 aggregations
        df_out['card1_amt_mean'] = df_out['card1'].map(self.card1_amt_mean)
        df_out['card1_amt_std'] = df_out['card1'].map(self.card1_amt_std)
        df_out['card1_amt_diff'] = df_out['TransactionAmt'] - df_out['card1_amt_mean']
        
        # card1_addr1 aggregations
        df_out['card1_addr1_amt_mean'] = df_out['card1_addr1'].map(self.card1_addr1_amt_mean)
        df_out['card1_addr1_amt_std'] = df_out['card1_addr1'].map(self.card1_addr1_amt_std)
        df_out['card1_addr1_amt_diff'] = df_out['TransactionAmt'] - df_out['card1_addr1_amt_mean']
        
        # Fill missing aggregation values with standard defaults
        df_out['card1_amt_mean'] = df_out['card1_amt_mean'].fillna(df_out['TransactionAmt'])
        df_out['card1_amt_std'] = df_out['card1_amt_std'].fillna(0)
        df_out['card1_amt_diff'] = df_out['card1_amt_diff'].fillna(0)
        df_out['card1_addr1_amt_mean'] = df_out['card1_addr1_amt_mean'].fillna(df_out['TransactionAmt'])
        df_out['card1_addr1_amt_std'] = df_out['card1_addr1_amt_std'].fillna(0)
        df_out['card1_addr1_amt_diff'] = df_out['card1_addr1_amt_diff'].fillna(0)
        
        return df_out
        
    def fit(self, X, y=None):
        # Prevent fitting on labels or ID columns
        cols_to_drop = ['TransactionID', 'isFraud', 'TransactionDT']
        X_clean = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        
        # Compute group-based statistical aggregations on train
        card1_stats = X_clean.groupby('card1')['TransactionAmt'].agg(['mean', 'std'])
        self.card1_amt_mean = card1_stats['mean'].to_dict()
        self.card1_amt_std = card1_stats['std'].to_dict()
        
        X_clean['card1_addr1'] = X_clean['card1'].astype(str) + "_" + X_clean['addr1'].astype(str)
        card1_addr1_stats = X_clean.groupby('card1_addr1')['TransactionAmt'].agg(['mean', 'std'])
        self.card1_addr1_amt_mean = card1_addr1_stats['mean'].to_dict()
        self.card1_addr1_amt_std = card1_addr1_stats['std'].to_dict()
        
        # Generate engineered features
        X_eng = self._create_features(X)
        X_eng = X_eng.drop(columns=[c for c in cols_to_drop if c in X_eng.columns])
        
        # Identify columns
        self.categorical_cols = list(X_eng.select_dtypes(include=['object', 'category']).columns)
        self.numerical_cols = list(X_eng.select_dtypes(exclude=['object', 'category']).columns)
        
        # Fit imputers
        self.num_imputer.fit(X_eng[self.numerical_cols])
        self.cat_imputer.fit(X_eng[self.categorical_cols])
        
        # Preprocess temporary matrices to do feature selection
        X_num_imp = self.num_imputer.transform(X_eng[self.numerical_cols])
        X_cat_imp = self.cat_imputer.transform(X_eng[self.categorical_cols])
        
        self.scaler.fit(X_num_imp)
        self.encoder.fit(X_cat_imp)
        
        X_num_scaled = self.scaler.transform(X_num_imp)
        X_cat_enc = self.encoder.transform(X_cat_imp)
        
        X_proc = np.hstack([X_num_scaled, X_cat_enc])
        all_features = self.numerical_cols + self.categorical_cols
        
        # Feature Selection using a quick LightGBM model
        if y is not None:
            print("Running baseline LightGBM feature selection...")
            dtrain = lgb.Dataset(X_proc, label=y)
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.train(params, dtrain)
            importances = model.feature_importance(importance_type='gain')
            indices = np.argsort(importances)[::-1]
            
            # Select top K
            selected_indices = indices[:self.top_k_features]
            self.selected_features = [all_features[i] for i in selected_indices]
            print(f"Selected top {len(self.selected_features)} features.")
        else:
            self.selected_features = all_features
            
        self.feature_names_out_ = self.selected_features
        self.fitted = True
        return self
        
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Transformer not fitted yet.")
            
        cols_to_drop = ['TransactionID', 'isFraud', 'TransactionDT']
        
        # Generate features
        X_eng = self._create_features(X)
        
        # Impute and encode
        X_num_imp = self.num_imputer.transform(X_eng[self.numerical_cols])
        X_cat_imp = self.cat_imputer.transform(X_eng[self.categorical_cols])
        
        X_num_scaled = self.scaler.transform(X_num_imp)
        X_cat_enc = self.encoder.transform(X_cat_imp)
        
        # Create a single dataframe to slice features easily
        df_num = pd.DataFrame(X_num_scaled, columns=self.numerical_cols)
        df_cat = pd.DataFrame(X_cat_enc, columns=self.categorical_cols)
        df_all = pd.concat([df_num, df_cat], axis=1)
        
        # Return only the selected features
        return df_all[self.selected_features]

def save_preprocessor(transformer, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(transformer, f)
    print(f"Saved preprocessor to {filepath}")

def load_preprocessor(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
