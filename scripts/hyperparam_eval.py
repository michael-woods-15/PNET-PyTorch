import logging
import sys
import os
import json
import statistics as stats
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from models.pnet import PNet
from training.pnet_trainer import PNetTrainer
from scripts.scripts_utils import set_random_seed

class HyperparameterEvaluator:
    def __init__(self, trials_per_config, configs_path, output_path, random_seed):
        self.trials_per_config = trials_per_config
        self.configs_path = configs_path
        self.output_path = output_path
        self.random_seed = random_seed

        set_random_seed(self.random_seed)

        logging.info("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = run_data_pipeline()
        self.connectivity_maps = get_connectivity_maps()[:5]

        self.eval_results = []


    def load_configs(self):
        with open(self.configs_path, 'r') as file:
            data = json.load(file) 

        return data


    def run_evaluations(self):
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting Evaluation of top 10 hyperparameter configurations")
        logging.info(f"Number of trials per config: {self.trials_per_config}")
        logging.info(f"Random seed: {self.random_seed}")
        logging.info(f"{'='*80}\n")

        hyperparam_configs = self.load_configs()

        for config_idx, config in enumerate(hyperparam_configs):
            logging.info(f"\nEvlauting Configuration No: {config_idx}")
            logging.info(f"Optuna Search Trial {config['number']}")
            logging.info(f"{json.dumps(config['params'], indent=2)}")
            val_losses = []

            for trial_idx in range(self.trials_per_config):
                trial_seed = self.random_seed + (config_idx * self.trials_per_config + trial_idx)
                set_random_seed(trial_seed)

                logging.info(f"\nTrial {trial_idx}: Testing configuration (seed: {trial_seed}):")
                val_loss, final_epoch, stopped_early = self.run_individual_trial(config["params"], config_idx, trial_idx)
                val_losses.append(val_loss)
            
            mean_val_loss = stats.mean(val_losses)
            std_val_loss = stats.stdev(val_losses)

            result = {
                "number": config["number"],
                "params": config["params"],
                "val_losses": val_losses,
                "mean_val_loss": mean_val_loss,
                "std_val_loss": std_val_loss

            }
            
            self.eval_results.append(result)

            logging.info(f"Config {config.get('number', config_idx)}: {config['params']}")
            logging.info(f"Validation Losses: {val_losses}")
            logging.info(f"Mean Loss: {mean_val_loss:.4f}, Std Dev: {std_val_loss:.4f}\n")


    def run_individual_trial(self, params, config_idx, trial_idx):
        model = PNet(
            connectivity_maps=self.connectivity_maps,
            n_genes=9229,
            n_modalities=3,
            dropout_h0=params['dropout_h0'],
            dropout_h=params['dropout_h']
        )
        
        trainer = PNetTrainer(
            model,
            self.train_loader,
            self.val_loader,
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            step_size=params['step_size'],
            gamma=params['gamma'],
            loss_weights=[2, 7, 20, 54, 148, 400],
            patience=20
        )

        trainer.train(n_epochs=300)
        
        best_val_loss = trainer.best_val_loss
        logging.info(f"Config {config_idx} Trial {trial_idx} completed with val_loss: {best_val_loss:.4f}")
        
        return best_val_loss, trainer.current_epoch, trainer.stopped_early
    

    def save_results(self):        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_filename = f"{self.output_path}/eval_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(self.eval_results, f, indent=2)
        logging.info(f"Results saved to JSON: {json_filename}")
        
        csv_filename = f"{self.output_path}/eval_results_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                'config_number', 'learning_rate', 'weight_decay', 'step_size', 
                'gamma', 'dropout_h0', 'dropout_h', 'mean_val_loss', 'std_val_loss',
                'trial_0', 'trial_1', 'trial_2', 'trial_3', 'trial_4'
            ])
            
            for result in self.eval_results:
                params = result['params']
                row = [
                    result['number'],
                    params['learning_rate'],
                    params['weight_decay'],
                    params['step_size'],
                    params['gamma'],
                    params['dropout_h0'],
                    params['dropout_h'],
                    f"{result['mean_val_loss']:.6f}",
                    f"{result['std_val_loss']:.6f}"
                ] + [f"{loss:.6f}" for loss in result['val_losses']]
                writer.writerow(row)
        
        logging.info(f"Results saved to CSV: {csv_filename}")
        return json_filename, csv_filename


    def report_summary(self):
        logging.info(f"\n{'='*80}")
        logging.info("EVALUATION SUMMARY")
        logging.info(f"{'='*80}\n")
        
        sorted_results = sorted(self.eval_results, key=lambda x: x['mean_val_loss'])
        
        logging.info(f"Total configurations evaluated: {len(self.eval_results)}")
        logging.info(f"Trials per configuration: {self.trials_per_config}")
        logging.info(f"Random seed: {self.random_seed}\n")
        
        logging.info("TOP 3 CONFIGURATIONS BY MEAN VALIDATION LOSS:")
        logging.info("-" * 80)
        for rank, result in enumerate(sorted_results[:3], 1):
            logging.info(f"\nRank {rank}: Configuration #{result['number']}")
            logging.info(f"  Mean Val Loss: {result['mean_val_loss']:.6f} ± {result['std_val_loss']:.6f}")
            logging.info(f"  Parameters:")
            for param, value in result['params'].items():
                logging.info(f"    {param}: {value}")
            logging.info(f"  Individual trial losses: {[f'{loss:.4f}' for loss in result['val_losses']]}")
        
        best_config = sorted_results[0]
        logging.info(f"\n{'='*80}")
        logging.info("BEST CONFIGURATION:")
        logging.info(f"  Config Number: {best_config['number']}")
        logging.info(f"  Mean Val Loss: {best_config['mean_val_loss']:.6f}")
        logging.info(f"  Std Dev: {best_config['std_val_loss']:.6f}")
        logging.info(f"  Best single trial: {min(best_config['val_losses']):.6f}")
        logging.info(f"  Worst single trial: {max(best_config['val_losses']):.6f}")
        
        all_means = [r['mean_val_loss'] for r in self.eval_results]
        logging.info(f"\n{'='*80}")
        logging.info("OVERALL STATISTICS:")
        logging.info(f"  Best mean val loss: {min(all_means):.6f}")
        logging.info(f"  Worst mean val loss: {max(all_means):.6f}")
        logging.info(f"  Average mean val loss: {stats.mean(all_means):.6f}")
        logging.info(f"  Std dev of means: {stats.stdev(all_means):.6f}")
        logging.info(f"{'='*80}\n")



if __name__ == '__main__':
    hyperparam_evaluator = HyperparameterEvaluator(
        trials_per_config=5,
        configs_path="../experiments/hyperparameters/optuna_search_final/top_10_configs.json",
        output_path="../checkpoints/",
        random_seed=242
    )

    hyperparam_evaluator.run_evaluations()
    hyperparam_evaluator.save_results()
    hyperparam_evaluator.report_summary()
    

