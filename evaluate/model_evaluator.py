import logging
import sys
import os 
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.pnet import PNet
from models.reactome_gnn import ReactomeGNN
from models.baseline import DenseNN
from training.metrics import MetricsTracker

class ModelEvaluator:
    def __init__(self, model_configs, test_loader, connectivity_maps):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_configs = model_configs
        self.test_loader = test_loader
        self.connectivity_maps = connectivity_maps  
        self.metrics = MetricsTracker(device=self.device)


    def load_model_checkpoint(self, model, model_name):
        path = os.path.join(os.getcwd(), f"../checkpoints/{model_name}_best_model.pt")
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    

    def evaluate_pnet(self, model, output_layer_index=None):
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
            'preds': (all_probs >= 0.5).long()
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

        pnet_output_layers = [None, 0, 1, 2, 3, 4, 5]
        for output_layer_idx in pnet_output_layers:
            results = self.evaluate_pnet(
                pnet_model,
                output_layer_idx
            )

            if output_layer_idx is None:
                all_results["PNet"] = results
            else:
                all_results[f"PNetSingle_output_layer_{output_layer_idx}"] = results

        # Reactome GNN evaluation
        gnn_model = ReactomeGNN(
            connectivity_maps=self.connectivity_maps,
            n_genes=9229,
            n_modalities=3,
            projection_dim = self.model_configs["ReactomeGNN"]["projection_dim"],
            hidden_dim = self.model_configs["ReactomeGNN"]["hidden_dim"],
            dropout_h0 = self.model_configs["ReactomeGNN"]["dropout_h0"],
            dropout = self.model_configs["ReactomeGNN"]["dropout_h"]
        )
        gnn_model = self.load_model_checkpoint(gnn_model, "gnn")

        gnn_results = self.evaluate_model(gnn_model)
        all_results["ReactomeGNN"] = gnn_results
    
        # Baseline Dense Neural Network evaluation
        baseline_model = model = DenseNN(
            n_genes=9229, 
            n_modalities=3,  
            dropout = self.model_configs["Baseline"]['dropout'],
            hidden_layers = self.model_configs["Baseline"]['hidden_layers']
        )
        baseline_model = self.load_model_checkpoint(baseline_model, "dense")

        baseline_results = self.evaluate_model(baseline_model)
        all_results["Baseline"] = baseline_results

        self.save_results(all_results)

    
    def save_results(self, all_results, filename="evaluation_results.json"):
        save_dict = {}

        for model_name, results in all_results.items():
            save_dict[model_name] = {
                "metrics": {
                    k: float(v) for k, v in results["metrics"].items()
                },
                "probs": results["probs"].tolist(),
                "preds": results["preds"].tolist()
            }

        save_path = os.path.join(os.getcwd(), filename)

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4)

        print(f"Results saved to {save_path}")



