import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def sample_dataset():
    data_dir = "d:/Advanced Financial Fraud Detection System/data"
    os.makedirs(data_dir, exist_ok=True)
    
    train_trans_path = os.path.join(data_dir, "train_transaction.csv")
    train_ident_path = os.path.join(data_dir, "train_identity.csv")
    test_trans_path = os.path.join(data_dir, "test_transaction.csv")
    test_ident_path = os.path.join(data_dir, "test_identity.csv")
    
    print("Starting chunk-based sampling of training transactions...")
    
    # Target totals for our sample:
    # We want a sample of 100,000 rows, preserving the ~3.5% fraud rate.
    # Target: 3,500 fraud rows, 96,500 non-fraud rows.
    target_fraud = 3500
    target_non_fraud = 96500
    
    sampled_fraud_dfs = []
    sampled_non_fraud_dfs = []
    
    total_fraud_collected = 0
    total_non_fraud_collected = 0
    
    chunksize = 50000
    # Read train transactions in chunks
    for chunk in pd.read_csv(train_trans_path, chunksize=chunksize):
        fraud_chunk = chunk[chunk['isFraud'] == 1]
        non_fraud_chunk = chunk[chunk['isFraud'] == 0]
        
        # Determine how many to sample from this chunk
        # IEEE-CIS has 590,540 rows in total (~12 chunks of 50k)
        # We sample proportionally from each chunk to preserve chronological sequence
        n_fraud_to_sample = min(len(fraud_chunk), int(np.ceil(target_fraud / 12.0)))
        n_non_fraud_to_sample = min(len(non_fraud_chunk), int(np.ceil(target_non_fraud / 12.0)))
        
        if n_fraud_to_sample > 0:
            sampled_fraud_dfs.append(fraud_chunk.sample(n=n_fraud_to_sample, random_state=42))
            total_fraud_collected += n_fraud_to_sample
            
        if n_non_fraud_to_sample > 0:
            sampled_non_fraud_dfs.append(non_fraud_chunk.sample(n=n_non_fraud_to_sample, random_state=42))
            total_non_fraud_collected += n_non_fraud_to_sample
            
    print(f"Collected {total_fraud_collected} fraud rows and {total_non_fraud_collected} non-fraud rows.")
    
    # Combine and shuffle
    trans_sampled = pd.concat(sampled_fraud_dfs + sampled_non_fraud_dfs, axis=0)
    trans_sampled = trans_sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Load identity file in chunks and merge on matching TransactionIDs
    print("Reading and merging training identity data...")
    sampled_ids = set(trans_sampled['TransactionID'])
    identity_chunks = []
    for chunk in pd.read_csv(train_ident_path, chunksize=chunksize):
        matching_rows = chunk[chunk['TransactionID'].isin(sampled_ids)]
        if not matching_rows.empty:
            identity_chunks.append(matching_rows)
            
    if identity_chunks:
        ident_sampled = pd.concat(identity_chunks, axis=0)
        # Left join to preserve all transactions
        train_full_sampled = trans_sampled.merge(ident_sampled, on='TransactionID', how='left')
    else:
        train_full_sampled = trans_sampled
        
    print(f"Merged training dataset shape: {train_full_sampled.shape}")
    
    # Split into train (80,000) and validation (20,000)
    y = train_full_sampled['isFraud']
    train_df, val_df = train_test_split(train_full_sampled, test_size=0.2, stratify=y, random_state=42)
    
    # Save sampled train and val
    train_out_path = os.path.join(data_dir, "train_sampled.csv")
    val_out_path = os.path.join(data_dir, "val_sampled.csv")
    train_df.to_csv(train_out_path, index=False)
    val_df.to_csv(val_out_path, index=False)
    print(f"Saved {len(train_df)} training samples to {train_out_path}")
    print(f"Saved {len(val_df)} validation samples to {val_out_path}")
    
    # Now let's also sample a smaller test set for inference and dashboard deployment
    print("Sampling test transactions...")
    test_sampled_dfs = []
    test_target = 20000
    for chunk in pd.read_csv(test_trans_path, chunksize=chunksize):
        n_test_to_sample = min(len(chunk), int(np.ceil(test_target / 11.0)))
        test_sampled_dfs.append(chunk.sample(n=n_test_to_sample, random_state=42))
        
    test_trans_sampled = pd.concat(test_sampled_dfs, axis=0).reset_index(drop=True)
    
    print("Reading and merging test identity data...")
    sampled_test_ids = set(test_trans_sampled['TransactionID'])
    test_identity_chunks = []
    for chunk in pd.read_csv(test_ident_path, chunksize=chunksize):
        matching_rows = chunk[chunk['TransactionID'].isin(sampled_test_ids)]
        if not matching_rows.empty:
            test_identity_chunks.append(matching_rows)
            
    if test_identity_chunks:
        test_ident_sampled = pd.concat(test_identity_chunks, axis=0)
        test_full_sampled = test_trans_sampled.merge(test_ident_sampled, on='TransactionID', how='left')
    else:
        test_full_sampled = test_trans_sampled
        
    test_out_path = os.path.join(data_dir, "test_sampled.csv")
    test_full_sampled.to_csv(test_out_path, index=False)
    print(f"Saved {len(test_full_sampled)} test samples to {test_out_path}")
    print("Sampling complete!")

if __name__ == "__main__":
    sample_dataset()
