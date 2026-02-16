import logging
import sys
import os  
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from models.pnet import PNet
from models.reactome_gnn import ReactomeGNN
from models.baseline import DenseNN
from training.pnet_trainer import PNetTrainer
from training.reactome_gnn_trainer import ReactomeGNNTrainer
from training.pnet_single_trainer import SingleOutputPNetTrainer
from training.baseline_trainer import DenseNNTrainer

def main(selected_model):
    train_loader, val_loader, test_loader = run_data_pipeline()
    connectivity_maps = get_connectivity_maps()[:5]

    if selected_model == "pnet":
        model = PNet(
            connectivity_maps=connectivity_maps, 
            n_genes = 9229, 
            n_modalities = 3, 
            dropout_h0=0.31161704591277983, 
            dropout_h=0.05102338209854125
        )

        trainer = PNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=0.009541006320883078,
            weight_decay=0.00011531477941868117,
            step_size=30,
            gamma=0.6078038714649643,
            loss_weights=[2, 7, 20, 54, 148, 400],
            patience=20
        )
    elif selected_model == "reactome_gnn":
        model = ReactomeGNN(
            connectivity_maps=connectivity_maps,
            n_genes=9229,
            n_modalities=3,
            projection_dim=32,
            hidden_dim=64,
            dropout_h0=0.5,
            dropout=0.3
        )

        trainer = ReactomeGNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=0.0005,
            weight_decay=0.01,
            step_size=30,
            gamma=0.5,
            patience=25,
        )
    elif selected_model == "pnet_single":
        model = PNet(
            connectivity_maps=connectivity_maps, 
            n_genes=9229, 
            n_modalities=3, 
            dropout_h0=0.31161704591277983, 
            dropout_h=0.05102338209854125
        )

        trainer = SingleOutputPNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=0.009541006320883078,
            weight_decay=0.00011531477941868117,
            step_size=30,
            gamma=0.6078038714649643,
            loss_weights=[2, 7, 20, 54, 148, 400],
            patience=20
        )
    elif selected_model == "dense":
        model = DenseNN(
            n_genes=9229, 
            n_modalities=3,  
            dropout=0.2,
            hidden_layers=4
        )

        trainer = DenseNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-4,
            weight_decay=0.001,
            step_size=20,
            gamma=0.75,         
            patience=25,
        )
    else:
        raise ValueError(
            f"Unknown model '{selected_model}'. "
            "Choose from ['pnet', 'reactome_gnn', 'pnet_single', 'dense']."
            )

    trainer.train(n_epochs=300, disable_early_stopping=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reactome-based models")
    parser.add_argument(
        "--selected_model",
        type=str,
        required=True,
        choices=["pnet", "reactome_gnn", "pnet_single", "dense"],
        help="Model to train"
    )

    args = parser.parse_args()
    main(args.selected_model)
