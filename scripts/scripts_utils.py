import torch
import logging
import sys
import os  
import random
import torch
import numpy as np

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings may reduce performance but increase reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed}")