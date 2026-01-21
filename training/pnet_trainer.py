import torch
from torch import optim
import torch.nn as nn
import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from training.losses import MultiOutputLoss
from training.metrics import MetricsTracker
from models.model_utils import count_parameters, save_model_checkpoint

class PNetTrainer:
    def __init__(self, model, train_loader, val_loader, lr, weight_decay, step_size, gamma, loss_weights, patience):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.loss_weights = loss_weights
        self.patience = patience

        self.optimiser = optim.Adam(
            model.parameters(),
            lr = self.learning_rate,
            weight_decay = self.weight_decay
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size = self.step_size,
            gamma = self.gamma  
        )

        self.loss_fn = MultiOutputLoss(loss_weights = self.loss_weights)
        self.metrics = MetricsTracker()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.min_delta = 1e-4
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.stopped_early = False

        total_params = count_parameters(self.model)
        logging.info(f"Device: {self.device}")
        logging.info(f"Model parameters: {total_params:,}")
        logging.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        #logging.info(f"Primary metric: {self.primary_metric}")

    def train_epoch(self):
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimiser.zero_grad()
            outputs = self.model(x)

            loss = self.loss_fn(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            total_loss += loss.item()

            avg_pred = torch.stack(outputs).mean(dim=0)
            self.metrics.update(avg_pred, y)

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics.compute()
        return avg_loss, metrics
    
    def validate(self):
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()

                avg_pred = torch.stack(outputs).mean(dim=0)
                self.metrics.update(avg_pred, y)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.compute()
        return avg_loss, metrics

    def train(self, n_epochs=300):
        logging.info(f"Starting training for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            logging.info(f"\nEpoch {epoch+1}/{n_epochs} | LR: {current_lr:.6f}")
            logging.info(f"  Train - Loss: {train_loss:.4f} | " + 
                        " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            logging.info(f"  Val   - Loss: {val_loss:.4f} | " + 
                        " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
            
            if val_loss < self.best_val_loss - self.min_delta:
                loss_improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                config = {'learning_rate': current_lr}
                save_model_checkpoint(self.model, self.optimiser, self.scheduler, epoch, val_loss, val_metrics, config)
                logging.info(f"    New best model saved (val_loss: {val_loss:.4f} - Improved by {loss_improvement})")
            else:
                self.epochs_without_improvement += 1
                logging.info(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            if self.epochs_without_improvement >= self.patience:
                self.stopped_early = True
                logging.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                logging.info(f"Best validation loss: {self.best_val_loss:.4f}")
                break

            
        logging.info("Training Completed")