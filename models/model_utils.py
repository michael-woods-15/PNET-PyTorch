import torch
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

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

    return total_params


def save_model_checkpoint(model, optimiser, scheduler, epoch, val_loss, val_metrics, config, model_type):
    """
    Save model checkpoint with optional training state.
    """

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_metrics': val_metrics,
        'config' : config
    }

    checkpoint_path = os.path.join(os.getcwd(), f'../checkpoints/{model_type}best_model.pt')
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def load_model_checkpoint(model, checkpoint_path, optimiser=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    info = {
        'epoch': checkpoint.get('epoch', 0),
        'val_loss': checkpoint.get('val_loss', None),
        'val_metrics': checkpoint.get('val_metrics', {}),
        'optimizer_loaded': False,
        'scheduler_loaded': False,
    }
    
    if optimiser is not None and 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        info['optimizsr_loaded'] = True
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        info['scheduler_loaded'] = True
    
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']

    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    logging.info(f"   Epoch: {info['epoch']}")
    if info['val_metrics']:
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in info['val_metrics'].items()])
        logging.info(f"    Validation metrics: {metrics_str}")
    if info['optimizer_loaded']:
        logging.info("    Optimizer state restored")
    if info['scheduler_loaded']:
        logging.info("    Scheduler state restored")
    
    return info
