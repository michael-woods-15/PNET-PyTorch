import logging
import sys
import os 
import yaml
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from scripts.scripts_utils import set_random_seed
from evaluate.train_final_models import FinalModelsTrainer
from evaluate.model_evaluator import ModelEvaluator
from evaluate.generate_plots import generate_all_plots


def load_model_configs():
        logging.info("Loading model configurations from YAML file")
        config_path = os.path.join(os.path.dirname(__file__), 'model_configs.yaml')
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        return {model['name']: model for model in configs['models']}


def evaluate_models():
    random_seed = 42
    set_random_seed(random_seed)

    model_configs = load_model_configs()

    train_loader, val_loader, test_loader = run_data_pipeline()
    connectivity_maps = get_connectivity_maps()[:5]

    logging.info("\nTraining final PNet, ReactomeGNN and Dense Baseline Models")
    trainer = FinalModelsTrainer(
        model_configs,
        train_loader,
        val_loader,
        connectivity_maps,
        random_seed
    )
    trainer.train_all_models()

    logging.info("\nEvaluating final PNet, Single Output PNet, ReactomeGNN and Dense Baseline Models")
    set_random_seed(random_seed) 
    evaluator = ModelEvaluator(
        model_configs,
        test_loader,
        connectivity_maps,
        random_seed
    )
    evaluator.evaluate_all_models()

    logging.info("\nGenerating plots of results")
    y_true = torch.cat([y for _, y in test_loader]).numpy()
    generate_all_plots(y_true)


if __name__ == '__main__':
    logging.info("Evaluating all Neural Network Models on Prostate Cancer Dataset")
    evaluate_models()

