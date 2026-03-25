import logging
import sys
import os 
import torch
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.pnet import PNet
from models.reactome_gnn import ReactomeGNN
from models.baseline import DenseNN
from training.metrics import MetricsTracker
from scripts.scripts_utils import set_random_seed

class ModelEvaluator:
    def __init__(self, model_configs, test_loader, connectivity_maps, random_seed):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_configs = model_configs
        self.test_loader = test_loader
        self.connectivity_maps = connectivity_maps  
        self.metrics = MetricsTracker(device=self.device)
        self.random_seed = random_seed


    def load_model_checkpoint(self, model, model_name):
        path = os.path.join(os.getcwd(), f"../checkpoints/{model_name}_best_model.pt")
    
        checkpoint = torch.load(path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        return model
    

    def evaluate_pnet(self, model, output_layer_index=None, pred_threshold=0.35):
        """
        Evaluates PNet or SingleOutputPNet on the test set.
        output_layer_index: if None, averages all outputs (standard PNet)
                            if int, uses that single output (SingleOutputPNet)
        """
        model.to(self.device)
        model.eval()
        self.metrics.reset()

        probs = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs = model(x)

                if output_layer_index is not None:
                    prob = torch.sigmoid(outputs[output_layer_index])
                else:
                    prob = torch.stack([torch.sigmoid(o) for o in outputs], dim=0).mean(dim=0)

                self.metrics.update(prob, y)
                probs.append(prob.cpu())

        all_probs = torch.cat(probs, dim=0)
        computed_metrics = self.metrics.compute()

        return {
            'metrics': computed_metrics,
            'probs': all_probs,
            'preds': (all_probs >= pred_threshold).long()
        }


    def evaluate_model(self, model):
        model.to(self.device)
        model.eval()
        self.metrics.reset()
    
        probs = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = model(x)
                prob = torch.sigmoid(logits)
                self.metrics.update(prob, y)
                probs.append(prob.cpu())

        all_probs = torch.cat(probs, dim=0)
        computed_metrics = self.metrics.compute()

        return {
            'metrics': computed_metrics,
            'probs': all_probs,
            'preds': (all_probs >= 0.5).long()
        }


    def evaluate_all_models(self):
        all_results = {}

        # Load PNet model once for the original and single outputs evaluation
        pnet_model = PNet(
            connectivity_maps = self.connectivity_maps, 
            n_genes=9229, 
            n_modalities=3, 
            dropout_h0 = self.model_configs["PNet"]['dropout_h0'], 
            dropout_h = self.model_configs["PNet"]['dropout_h']
        )
        pnet_model = self.load_model_checkpoint(pnet_model, "pnet")

        PNET_PREDICTION_THRESHOLD = 0.35
        self.metrics.set_threshold(PNET_PREDICTION_THRESHOLD)
        pnet_output_layers = [None, 0, 1, 2, 3, 4, 5]
        for output_layer_idx in pnet_output_layers:
            set_random_seed(self.random_seed)
            results = self.evaluate_pnet(
                pnet_model,
                output_layer_idx,
                PNET_PREDICTION_THRESHOLD
            )

            if output_layer_idx is None:
                all_results["PNet"] = results
                logging.info("\nPNet results")
                logging.info(f"F1: {results['metrics']["f1"]}")
                logging.info(f"AUC: {results['metrics']["auc"]}")
                logging.info(f"Accuracy: {results['metrics']["accuracy"]}")
                logging.info(f"Precision: {results['metrics']["precision"]}")
                logging.info(f"Recall: {results['metrics']["recall"]}")
                logging.info(f"Cohen Kappa: {results['metrics']["cohen_kappa"]}")
            else:
                all_results[f"PNetSingle_output_layer_{output_layer_idx}"] = results
                logging.info(f"\nPNet Single Output {output_layer_idx} results:")
                logging.info(f"F1: {results['metrics']["f1"]}")
                logging.info(f"AUC: {results['metrics']["auc"]}")
                logging.info(f"Accuracy: {results['metrics']["accuracy"]}")
                logging.info(f"Precision: {results['metrics']["precision"]}")
                logging.info(f"Recall: {results['metrics']["recall"]}")
                logging.info(f"Cohen Kappa: {results['metrics']["cohen_kappa"]}")

        # Reactome GNN evaluation
        self.metrics.set_threshold(0.5)
        gnn_model = ReactomeGNN(
            connectivity_maps=self.connectivity_maps,
            n_genes=9229,
            n_modalities=3,
            projection_dim = self.model_configs["ReactomeGNN"]["projection_dim"],
            hidden_dim = self.model_configs["ReactomeGNN"]["hidden_dim"],
            dropout_h0 = self.model_configs["ReactomeGNN"]["dropout_h0"],
            dropout = self.model_configs["ReactomeGNN"]["dropout_h"]
        )
        gnn_model = self.load_model_checkpoint(gnn_model, "reactome_gnn")

        set_random_seed(self.random_seed)
        gnn_results = self.evaluate_model(gnn_model)

        logging.info("\nGNN results")
        logging.info(f"F1: {gnn_results['metrics']["f1"]}")
        logging.info(f"AUC: {gnn_results['metrics']["auc"]}")
        logging.info(f"Accuracy: {gnn_results['metrics']["accuracy"]}")
        logging.info(f"Precision: {gnn_results['metrics']["precision"]}")
        logging.info(f"Recall: {gnn_results['metrics']["recall"]}")
        logging.info(f"Cohen Kappa: {gnn_results['metrics']["cohen_kappa"]}")

        all_results["ReactomeGNN"] = gnn_results
    
        # Baseline Dense Neural Network evaluation
        baseline_model  = DenseNN(
            n_genes=9229, 
            n_modalities=3,  
            dropout = self.model_configs["Baseline"]['dropout'],
            hidden_layers = self.model_configs["Baseline"]['hidden_layers']
        )
        baseline_model = self.load_model_checkpoint(baseline_model, "dense")

        set_random_seed(self.random_seed)
        baseline_results = self.evaluate_model(baseline_model)

        logging.info("\nBaseline results")
        logging.info(f"F1: {baseline_results['metrics']["f1"]}")
        logging.info(f"AUC: {baseline_results['metrics']["auc"]}")
        logging.info(f"Accuracy: {baseline_results['metrics']["accuracy"]}")
        logging.info(f"Precision: {baseline_results['metrics']["precision"]}")
        logging.info(f"Recall: {baseline_results['metrics']["recall"]}")
        logging.info(f"Cohen Kappa: {baseline_results['metrics']["cohen_kappa"]}")

        all_results["Baseline"] = baseline_results

        self.save_results(all_results)

    
    def save_results(self, all_results, filename="evaluation_results.json"):
        save_dict = {}

        for model_name, results in all_results.items():
            save_dict[model_name] = {
                "metrics": {
                    k: float(v) for k, v in results["metrics"].items()
                },
                "probs": results["probs"].squeeze().tolist(),
                "preds": results["preds"].squeeze().tolist()
            }

        save_path = os.path.join(os.getcwd(), filename)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=2)

        print(f"Results saved to {save_path}")



