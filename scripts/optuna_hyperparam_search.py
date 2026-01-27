import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
import optuna
from optuna.trial import TrialState
import csv
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from data_access.data_pipeline import run_data_pipeline
from reactome.pathway_hierarchy import get_connectivity_maps
from models.pnet import PNet
from training.pnet_trainer import PNetTrainer

def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings may reduce performance but increase reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed}")


class OptunaHyperparameterSearch:
    def __init__(self, n_trials=50, results_dir='../experiments/hyperparameters', random_seed=42):
        self.n_trials = n_trials
        self.results_dir = Path(results_dir)
        self.random_seed = random_seed

        set_random_seed(self.random_seed)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.results_dir / f'optuna_search_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Loading data...")
        self.train_loader, self.val_loader, self.test_loader = run_data_pipeline()
        self.connectivity_maps = get_connectivity_maps()[:5]
        
        self.study_name = f'pnet_optimization_{timestamp}'
        storage_name = f"sqlite:///{self.run_dir / 'optuna_study.db'}"

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=40,  
            interval_steps=10  
        )
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_name,
            direction='minimize',
            pruner=pruner,
            load_if_exists=False
        )


    def objective(self, trial):
        trial_seed = self.random_seed + trial.number
        set_random_seed(trial_seed)

        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'step_size': trial.suggest_int('step_size', 30, 70, step=10),
            'gamma': trial.suggest_float('gamma', 0.5, 0.95),
            'dropout_h0': trial.suggest_float('dropout_h0', 0.3, 0.7),
            'dropout_h': trial.suggest_float('dropout_h', 0.0, 0.2),
            
            # Fixed parameters
            'max_epochs': 300,
            'patience': 20,
            
            # Categorical for loss_weights (uncomment if needed)
            # 'loss_weights': trial.suggest_categorical('loss_weights', [
            #     [4, 14, 25, 43, 74, 200],
            #     [2, 7, 20, 54, 148, 400],
            #     [1, 4, 10, 65, 222, 600]
            # ])
        }

        logging.info(f"\nTrial {trial.number}: Testing configuration (seed: {trial_seed}):")
        logging.info(f"{json.dumps(config, indent=2)}")

        try:
            model = PNet(
                connectivity_maps=self.connectivity_maps,
                n_genes=9229,
                n_modalities=3,
                dropout_h0=config['dropout_h0'],
                dropout_h=config['dropout_h']
            )
            
            trainer = PNetTrainer(
                model,
                self.train_loader,
                self.val_loader,
                lr=config['learning_rate'],
                weight_decay=config['weight_decay'],
                step_size=config['step_size'],
                gamma=config['gamma'],
                loss_weights=config.get('loss_weights', [2, 7, 20, 54, 148, 400]),
                patience=config['patience']
            )

            trainer.train(n_epochs=config['max_epochs'], optuna_trial=trial)

            trial.set_user_attr('final_epoch', trainer.current_epoch)
            trial.set_user_attr('stopped_early', trainer.stopped_early)
            trial.set_user_attr('trial_seed', trial_seed)
            
            best_val_loss = trainer.best_val_loss
            logging.info(f"Trial {trial.number} completed with val_loss: {best_val_loss:.4f}")
            
            return best_val_loss
        
        except Exception as e:
            logging.error(f"Trial {trial.number} failed with error: {str(e)}")
            raise optuna.TrialPruned()
        

    def run_search(self):
        logging.info(f"\n{'='*80}")
        logging.info(f"Starting Optuna Hyperparameter Search")
        logging.info(f"Number of trials: {self.n_trials}")
        logging.info(f"Random seed: {self.random_seed}")
        logging.info(f"{'='*80}\n")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        self.save_results()
        self.print_summary()


    def save_results(self):
        # Save study statistics
        study_stats = {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial_number': self.study.best_trial.number,
            'random_seed': self.random_seed,
            'datetime_start': self.study.trials[0].datetime_start.isoformat() if self.study.trials else None,
            'datetime_complete': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'study_summary.json', 'w') as f:
            json.dump(study_stats, f, indent=2)
        
        # Save all trials
        trials_data = []
        for trial in self.study.trials:
            if trial.state == TrialState.COMPLETE:
                trial_dict = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs,
                    'state': trial.state.name
                }
                trials_data.append(trial_dict)
        
        with open(self.run_dir / 'all_trials.json', 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Save top 10 trials
        completed_trials = [t for t in trials_data]
        sorted_trials = sorted(completed_trials, key=lambda x: x['value'])
        top_10 = sorted_trials[:10]
        
        with open(self.run_dir / 'top_10_configs.json', 'w') as f:
            json.dump(top_10, f, indent=2)
        
        # Save as CSV
        if completed_trials:
            csv_file = self.run_dir / 'results_summary.csv'
            param_keys = list(completed_trials[0]['params'].keys())
            attr_keys = list(completed_trials[0]['user_attrs'].keys()) if completed_trials[0]['user_attrs'] else []
            
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['trial_number', 'val_loss'] + param_keys + attr_keys
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for trial in sorted_trials:
                    row = {
                        'trial_number': trial['number'],
                        'val_loss': trial['value']
                    }
                    row.update(trial['params'])
                    row.update(trial['user_attrs'])
                    writer.writerow(row)

        try:
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.savefig(self.run_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.savefig(self.run_dir / 'param_importances.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Saved optimization plots")
        except Exception as e:
            logging.warning(f"Could not generate plots: {str(e)}")


    def print_summary(self):
        logging.info(f"\n{'='*80}")
        logging.info("OPTUNA OPTIMIZATION SUMMARY")
        logging.info(f"{'='*80}")
        
        pruned_trials = [t for t in self.study.trials if t.state == TrialState.PRUNED]
        complete_trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        failed_trials = [t for t in self.study.trials if t.state == TrialState.FAIL]
        
        logging.info(f"Total trials: {len(self.study.trials)}")
        logging.info(f"  Completed: {len(complete_trials)}")
        logging.info(f"  Pruned: {len(pruned_trials)}")
        logging.info(f"  Failed: {len(failed_trials)}")
        
        if complete_trials:
            logging.info(f"\nBest trial: #{self.study.best_trial.number}")
            logging.info(f"  Best validation loss: {self.study.best_value:.4f}")
            logging.info(f"\nBest hyperparameters:")
            for key, value in self.study.best_params.items():
                logging.info(f"    {key}: {value}")
            
            # Show top 5 trials
            sorted_trials = sorted(complete_trials, key=lambda x: x.value)
            logging.info(f"\nTop 5 Trials:")
            for i, trial in enumerate(sorted_trials[:5], 1):
                logging.info(f"\n  {i}. Trial #{trial.number}")
                logging.info(f"     Val Loss: {trial.value:.4f}")
                logging.info(f"     Params: {json.dumps(trial.params, indent=13)}")
        
        logging.info(f"\nResults saved to: {self.run_dir}")
        logging.info(f"Study database: {self.run_dir / 'optuna_study.db'}")
        logging.info(f"Random seed: {self.random_seed}")
        logging.info(f"{'='*80}\n")


if __name__ == '__main__':
    searcher = OptunaHyperparameterSearch(n_trials=50)
    searcher.run_search()
