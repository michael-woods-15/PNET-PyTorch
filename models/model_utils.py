import torch.nn as nn
import torch
import logging
import sys
import os  
import tempfile
import shutil
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.pnet import PNet
from reactome.pathway_hierarchy import get_connectivity_maps

def count_parameters(model):
    total_params = 0
    total_trainable = 0
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        if module_params > 0:
            total_params += module_params
            total_trainable += module_trainable

    logging.info("=" * 80)
    logging.info(f"{'Total Parameters':50s} {total_params:>10,}")
    logging.info(f"{'Trainable Parameters':50s} {total_trainable:>10,}")
    logging.info(f"{'Non-trainable Parameters':50s} {total_params - total_trainable:>10,}")
    logging.info("=" * 80)


def save_model(model, path, include_optimizer=False, optimizer=None, epoch=None, loss=None):
    """
    Save model checkpoint with optional training state.
    """

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if include_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, path)
    logging.info(f"Model saved to {path}")


def load_model(model, path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    """

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    info = {}
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        info['optimizer_loaded'] = True
    
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
    
    if 'loss' in checkpoint:
        info['loss'] = checkpoint['loss']
    
    logging.info(f"Model loaded from {path}")
    if info:
        logging.info(f"   Checkpoint info: {info}")
    
    return info
