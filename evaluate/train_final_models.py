import logging
import sys
import os 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.pnet import PNet
from models.reactome_gnn import ReactomeGNN
from models.baseline import DenseNN
from training.pnet_trainer import PNetTrainer
from training.reactome_gnn_trainer import ReactomeGNNTrainer
from training.baseline_trainer import DenseNNTrainer

class FinalModelsTrainer:
    def __init__(self, model_configs, train_loader, val_loader, connectivity_maps):
        self.model_configs = model_configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.connectivity_maps = connectivity_maps


    def train_pnet(self, config):
        model = PNet(
            connectivity_maps = self.connectivity_maps, 
            n_genes=9229, 
            n_modalities=3, 
            dropout_h0 = config['dropout_h0'], 
            dropout_h = config['dropout_h']
        )

        trainer = PNetTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr = config['lr'],
            weight_decay = config['weight_decay'],
            step_size = config['step_size'],
            gamma = config['gamma'],
            loss_weights = config['loss_weights'],
            patience = config['patience']
        )

        trainer.train(n_epochs=config['max_epochs'])


    def train_baseline(self, config):
        model = DenseNN(
            n_genes=9229, 
            n_modalities=3,  
            dropout = config['dropout'],
            hidden_layers = config['hidden_layers']
        )

        trainer = DenseNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr = config['lr'],
            weight_decay = config['weight_decay'],
            step_size = config['step_size'],
            gamma = config['gamma'],         
            patience = config['patience'],
        )

        trainer.train(n_epochs=config['max_epochs'])


    def train_reactome_gnn(self, config):
        model = ReactomeGNN(
            connectivity_maps=self.connectivity_maps,
            n_genes=9229,
            n_modalities=3,
            projection_dim = config["projection_dim"],
            hidden_dim = config["hidden_dim"],
            dropout_h0 = config["dropout_h0"],
            dropout = config["dropout_h"]
        )

        trainer = ReactomeGNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            lr = config["learning_rate"],
            weight_decay = config["weight_decay"],
            pathway_weight_decay = config["pathway_weight_decay"],
            step_size = config["step_size"],
            gamma = config["gamma"],
            patience = config["patience"],
        )

        trainer.train(n_epochs=config['max_epochs'])


    def train_all_models(self):
        self.train_pnet(self.model_configs["PNet"])
        self.train_baseline(self.model_configs["Baseline"])
        self.train_reactome_gnn(self.model_configs["ReactomeGNN"])
