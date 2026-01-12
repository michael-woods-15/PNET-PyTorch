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
from models.model_utils import count_parameters

class PNetTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimiser = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.001
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimiser,
            step_size=50,
            gamma=0.75   # reduce LR by 25%
        )

        self.loss_fn = MultiOutputLoss()
        self.metrics = MetricsTracker()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

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
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            logging.info(f"\nEpoch {epoch+1}/{n_epochs} | LR: {current_lr:.6f}")
            logging.info(f"  Train - Loss: {train_loss:.4f} | " + 
                        " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            logging.info(f"  Val   - Loss: {val_loss:.4f} | " + 
                        " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
            
        logging.info("Training Completed")