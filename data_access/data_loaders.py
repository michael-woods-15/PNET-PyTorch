import logging
import pandas as pd
import sys
import os 
import torch
from torch.utils.data import TensorDataset, DataLoader 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

def create_dataloaders(x_train, x_val, x_test, y_train, y_val, y_test, 
                       batch_size=32, shuffle_train=True, num_workers=0):
    
    x_train_tensor = torch.FloatTensor(x_train.values)
    x_val_tensor = torch.FloatTensor(x_val.values)
    x_test_tensor = torch.FloatTensor(x_test.values)
    
    y_train_tensor = torch.FloatTensor(y_train)
    y_val_tensor = torch.FloatTensor(y_val)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader