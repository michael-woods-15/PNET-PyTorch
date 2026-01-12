import logging
import sys
import os  
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, BinaryAccuracy,
    BinaryPrecision, BinaryRecall, BinaryCohenKappa
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

class MetricsTracker:
    def __init__(self):
        self.f1 = BinaryF1Score()
        self.auc = BinaryAUROC()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.cohen_kappa = BinaryCohenKappa()

    def update(self, preds, targets):
        preds = preds.detach()
        targets = targets.detach()

        self.f1.update(preds, targets)
        self.auc.update(preds, targets)
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.cohen_kappa.update(preds, targets)

    def compute(self):
        return {
            'f1': self.f1.compute().item(),
            'auc': self.auc.compute().item(),
            'accuracy': self.accuracy.compute().item(),
            'precision': self.precision.compute().item(),
            'recall': self.recall.compute().item(),
            'cohen_kappa': self.cohen_kappa.compute().item()
        }

    def reset(self):
        self.f1.reset()
        self.auc.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.cohen_kappa.reset()
