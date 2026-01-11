import logging
import pandas as pd
import sys
import os 
import torch
from torch.utils.data import TensorDataset, DataLoader 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.genomic_data import run_genomic_data_pipeline
from data_access.data_splits import get_train_validate_test_splits
from data_access.data_loaders import create_dataloaders

def run_data_pipeline():
    features, responses, sample_ids = run_genomic_data_pipeline(
        use_selected_genes_only = True, 
        use_coding_genes_only = True, 
        combine_type = 'union'
    )

    x_train, x_val, x_test, y_train, y_val, y_test, train_sample_ids, val_sample_ids, test_sample_ids = \
          get_train_validate_test_splits(features, responses, sample_ids)

    print("Summary of data splits")
    print(f"x_train: {x_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"x_test: {x_test.shape}")

    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"y_test: {y_test.shape}")

    print(f"ids train: {train_sample_ids.shape}")
    print(f"ids val: {val_sample_ids.shape}")
    print(f"ids test: {test_sample_ids.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(
        x_train, x_val, x_test, 
        y_train, y_val, y_test, 
        batch_size=32, 
        shuffle_train=True, 
        num_workers=0
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    run_data_pipeline()