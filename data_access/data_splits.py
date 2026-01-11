import logging
import pandas as pd
import sys
import os 
import torch
from torch.utils.data import TensorDataset, DataLoader 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from genomic_data import run_genomic_data_pipeline

from config_path import DATA_SPLITS_PATH
TRAIN_SPLIT_FILE = os.path.join(DATA_SPLITS_PATH, 'training_set.csv')
VAL_SPLIT_FILE = os.path.join(DATA_SPLITS_PATH, 'validation_set.csv')
TEST_SPLIT_FILE = os.path.join(DATA_SPLITS_PATH, 'test_set.csv')

def get_train_validate_test_splits(features, responses, sample_ids):
    logging.info('Processing train/val/test splits')

    train_set_df = pd.read_csv(TRAIN_SPLIT_FILE)
    val_set_df = pd.read_csv(VAL_SPLIT_FILE)
    test_set_df = pd.read_csv(TEST_SPLIT_FILE)

    sample_id_set = set(sample_ids)#

    train_ids = sample_id_set.intersection(train_set_df.id)
    val_ids = sample_id_set.intersection(val_set_df.id)
    test_ids = sample_id_set.intersection(test_set_df.id)

    train_mask = sample_ids.isin(train_ids)
    val_mask = sample_ids.isin(val_ids)
    test_mask = sample_ids.isin(test_ids)

    x_train = features[train_mask]
    x_val = features[val_mask]
    x_test = features[test_mask]

    y_train = responses[train_mask].values.reshape(-1, 1)
    y_val = responses[val_mask].values.reshape(-1, 1)
    y_test = responses[test_mask].values.reshape(-1, 1)

    train_sample_ids = sample_ids[train_mask]
    val_sample_ids = sample_ids[val_mask]
    test_sample_ids = sample_ids[test_mask]

    return x_train, x_val, x_test, y_train, y_val, y_test, train_sample_ids, val_sample_ids, test_sample_ids

if __name__ == "__main__":
    features, responses, sample_ids = run_genomic_data_pipeline(use_selected_genes_only = True, use_coding_genes_only = True, combine_type = 'union')
    x_train, x_val, x_test, y_train, y_val, y_test, train_sample_ids, val_sample_ids, test_sample_ids = get_train_validate_test_splits(features, responses, sample_ids)


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
