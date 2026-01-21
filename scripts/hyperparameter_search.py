import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from itertools import product
import random
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from models.pnet import PNet
from training.pnet_trainer import PNetTrainer

class HyperparameterSearch:
    def __init__(self, search_space, n_trials, results_dir='../experiments/hyperparameters'):
        self.search_space = search_space
        self.n_trials = n_trials
        self.results_dir = Path(results_dir)
        self.results = []

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.results_dir / f'search_{timestamp}'
        self.run_dir.mkdir(exist_ok=True)


    def generate_configs(self):
        all_configs = [
            {k: v for k, v in zip(self.search_space.keys(), values)}
            for values in product(*self.search_space.values())
        ]
        
        configs = random.sample(all_configs, self.n_trials)
        return configs
    

    def run_single_trial(self, config, trial_num, total_trials, train_loader, val_loader, connectivity_maps):
        logging.info(f"\n{'='*80}")
        logging.info(f"Trial {trial_num}/{total_trials}")
        logging.info(f"Configuration: {json.dumps(config, indent=2)}")
        logging.info(f"{'='*80}\n")
        
        try:
            model = PNet(
                connectivity_maps=connectivity_maps,
                n_genes=9229,
                n_modalities=3,
                dropout_h0=config.get('dropout_h0', 0.5),
                dropout_h=config.get('dropout_h', 0.1)
            )
            
            trainer = PNetTrainer(
                model,
                train_loader,
                val_loader,
                lr=config.get('learning_rate', 0.001),
                weight_decay=config.get('weight_decay', 0.001),
                step_size=config.get('step_size', 50),
                gamma=config.get('gamma', 0.75),
                loss_weights=config.get('loss_weights', [2, 7, 20, 54, 148, 400]),
                patience=config.get('patience', 20)
            )
            
            trainer.train(n_epochs=config.get('max_epochs', 300))
            
            result = {
                'trial_num': trial_num,
                'config': config,
                'best_val_loss': trainer.best_val_loss,
                'final_epoch': trainer.current_epoch,
                'stopped_early': trainer.stopped_early
            }
            
            logging.info(f"\nTrial {trial_num} completed:")
            logging.info(f"  Best val loss: {result['best_val_loss']:.4f}")
            logging.info(f"  Final epoch: {result['final_epoch']}")
            
            return result
            
        except Exception as e:
            logging.error(f"Trial {trial_num} failed with error: {str(e)}")
            return {
                'trial_num': trial_num,
                'config': config,
                'error': str(e),
                'best_val_loss': float('inf')
            }


    def run_search(self):
        logging.info("Loading data...")
        train_loader, val_loader, test_loader = run_data_pipeline()
        connectivity_maps = get_connectivity_maps()[:5]
        
        configs = self.generate_configs()
        total_trials = len(configs)
        
        search_config = {
            'n_trials': total_trials,
            'search_space': self.search_space,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.run_dir / 'search_config.json', 'w') as f:
            json.dump(search_config, f, indent=2)
        
        for i, config in enumerate(configs, 1):
            result = self.run_single_trial(
                config, i, total_trials, 
                train_loader, val_loader, connectivity_maps
            )
            self.results.append(result)
        
        self.save_results()
        self.print_summary()


    def save_results(self):
        with open(self.run_dir / 'all_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        successful_results = [r for r in self.results if 'error' not in r]
        sorted_results = sorted(successful_results, key=lambda x: x['best_val_loss'])
        top_10 = sorted_results[:10]
        with open(self.run_dir / 'top_10_configs.json', 'w') as f:
            json.dump(top_10, f, indent=2)
        
        csv_file = self.run_dir / 'results_summary.csv'
        if successful_results:
            keys = ['trial_num', 'best_val_loss', 'final_epoch', 'stopped_early']
            config_keys = list(successful_results[0]['config'].keys())
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys + config_keys)
                writer.writeheader()
                for result in sorted_results:
                    row = {k: result[k] for k in keys}
                    row.update(result['config'])
                    writer.writerow(row)
    

    def print_summary(self):
        """Print summary of search results"""
        successful_results = [r for r in self.results if 'error' not in r]
        failed_trials = len(self.results) - len(successful_results)
        
        logging.info(f"\n{'='*80}")
        logging.info("HYPERPARAMETER SEARCH SUMMARY")
        logging.info(f"{'='*80}")
        logging.info(f"Total trials: {len(self.results)}")
        logging.info(f"Successful: {len(successful_results)}")
        logging.info(f"Failed: {failed_trials}")
        
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda x: x['best_val_loss'])
            
            logging.info(f"\nTop 5 Configurations:")
            for i, result in enumerate(sorted_results[:5], 1):
                logging.info(f"\n{i}. Val Loss: {result['best_val_loss']:.4f}")
                logging.info(f"   Config: {json.dumps(result['config'], indent=11)}")
        
        logging.info(f"\nResults saved to: {self.run_dir}")
        logging.info(f"{'='*80}\n")


def main():
    search_space = {
        'learning_rate': [0.0005, 0.001, 0.005],
        'weight_decay': [0.0005, 0.001, 0.005],
        'step_size': [40, 50, 60],
        'gamma': [0.6, 0.75, 0.9],
        'dropout_h0': [0.4, 0.5, 0.6],
        'dropout_h': [0.05, 0.1, 0.15],
        'max_epochs': [300],
        'patience': [20],
        #'loss_weights': [[4, 14, 25, 43, 74, 200], [2, 7, 20, 54, 148, 400], [1, 4, 10, 65, 222, 600]]
    }
    

    searcher = HyperparameterSearch(search_space, n_trials=3)
    searcher.run_search()


if __name__ == '__main__':
    main()