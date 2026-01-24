import logging
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from models.pnet import PNet
from training.pnet_trainer import PNetTrainer

def main():
    train_loader, val_loader, test_loader = run_data_pipeline()
    connectivity_maps = get_connectivity_maps()[:5]

    model = PNet(
        connectivity_maps = connectivity_maps, 
        n_genes = 9229, 
        n_modalities = 3, 
        dropout_h0 = 0.5, 
        dropout_h = 0.1
    )

    trainer = PNetTrainer(
        model,
        train_loader,
        val_loader,
        lr=0.001,
        weight_decay=0.001,
        step_size=50,
        gamma=0.75,
        loss_weights=[2, 7, 20, 54, 148, 400],
        patience=20
    )

    trainer.train(n_epochs=300)


if __name__ == '__main__':
    main()
