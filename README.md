# Biologically Informed Neural Networks for Interpretable Prostate Cancer Metastasis Prediction

## Overview
This repository contains the source code for my Level 4 MSci Computing Science project at the 
University of Glasgow, supervised by Dr. Jake Lever.

The project centres on a PyTorch reimplementation of P-NET, a biologically informed neural network 
(BINN) first introduced by Elmarakeby et al. (2021) for classifying prostate cancer samples as 
either primary, localised cancer or metastatic disease. P-NET is structured around a biological 
hierarchy derived from the Reactome pathway database, taking genomic input features — somatic 
mutations and copy number variants — as input.

This reimplementation was developed alongside two competing architectures: a biologically agnostic 
dense neural network and a novel Reactome-based graph neural network (GNN). Together, these models 
were used to investigate how P-NET performs relative to alternative approaches on the binary 
classification task of distinguishing primary from metastatic prostate cancer.

A dissertation detailing the research and findings accompanies this project. The abstract is 
included below, along with instructions for accessing the data and running the source code.

## Abstract
Prostate cancer is the most common cancer diagnosis among men in the UK, with progression to metastasis reducing the five-year survival rate from over 97\% to 53.2\%. Accurate detection of metastatic disease is therefore clinically crucial to ensure appropriate treatment and care. Deep learning applied to genomic data has shown promise for this task, but its lack of interpretability limits clinical applicability. Biologically informed neural networks (BINNs), such as P-NET, aim to address this by incorporating prior biological knowledge into model architecture. This study systematically evaluates P-NET to determine whether its architectural components meaningfully contribute to performance, and whether biologically informed approaches outperform conventional alternatives. 
    
P-NET was reimplemented in PyTorch and evaluated alongside a biologically agnostic dense model and a novel Reactome-based graph neural network. All models were trained and tested on a cohort of over 1,000 prostate cancer samples, with McNemar’s test applied to allow statistical comparison. P-NET achieved an AUROC of 0.945, matching a 117 million-parameter dense model with only 0.06\% of its parameters, demonstrating the effectiveness of the Reactome prior as an inductive bias. No individual output layer within P-NET significantly outperformed the full model and the gene-level output (Layer 0) achieved notably high recall (0.971), suggesting potential value in screening contexts. P-NET also significantly outperformed the Reactome GNN (AUROC 0.859, p = 0.0106), although limitations in the GNN design are acknowledged. Notably, four samples were misclassified by all models, indicating possible limits of genomic signal alone. Overall, BINNs such as P-NET provide a unique combination of interpretability, parameter efficiency, and strong predictive performance for metastasis prediction, although the clinical interpretability of learned representations remains an open question.

## Data Availability
All data used in this study, including the dataset and Reactome hierarchy, was made freely 
available by Elmarakeby et al. (2021) and can be accessed [here](https://zenodo.org/records/5163213).

The **'_database.zip'** folder should be downloaded from the above link, decompressed, and placed 
in the root directory of this repository. If any issues occur, ensure the filepaths in 
**'config_path.py'** are consistent with your local setup.

## Running the Code
### 1. Set Up Your Environment

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate        
pip install -r requirements.txt
```

**Using venv (Windows):**
```bash
python -m venv venv
call venv\Scripts\activate       
pip install -r requirements.txt
```

**Using Conda:**
```bash
conda create -n pnet python=3.11.5
conda activate pnet
pip install -r requirements.txt
```

### 2. Prepare the Data

Download the **'_database.zip'** folder from the link in the [Data Availability](#data-availability) 
section, decompress it, and place it in the root directory of this repository so that the structure 
looks as follows:
```
/
├── _database/
├── evaluate/
├── ...
```

### 3. Run the Experiments

Navigate to the `evaluate/` directory and run the evaluation script:
```bash
cd evaluate
python evaluate_all.py
```

## Project Structure
```
├── _database/          # Data folder (see Data Availability)
├── data_access/
├── evaluate/
├── models/
├── reactome/
├── scripts/
├── train/
└── config_path.py
```

### `data_access/`
Contains all dataset processing logic, including feature and response extraction, modality early 
integration, and gene filtering. Also includes PyTorch data loaders and train/test/validation 
splitting utilities. The full data processing pipeline can be run with:
```bash
python data_pipeline.py
```

### `evaluate/`
Contains the code for final experimental evaluation, including hyperparameter configurations for 
all three models and scripts to train and evaluate each final model. Also includes plotting 
utilities, statistical tests, and model agreement analysis. The full evaluation pipeline can be 
run with:
```bash
python evaluate_all.py
```

### `models/`
Contains the source code for all three model architectures — P-NET (and its single layer variants), the dense neural network, and 
the Reactome-based GNN — along with any custom layers implemented as part of this project.

### `reactome/`
Contains code for processing the Reactome pathway hierarchy to form the biologically informed 
basis of the BINN and GNN models. The full processing pipeline can be run with:
```bash
python pathway_hierarchy.py
```

### `scripts/`
Contains a collection of utility scripts for training individual models, running Optuna 
hyperparameter searches, and selecting candidate configurations produced by Optuna for each model.

### `train/`
Contains model trainers for all three architectures as well as single-layer variants of P-NET used 
throughout the project. Also includes a custom BCE loss implementation and a metrics tracker.

## References
- Elmarakeby, H. A., Hwang, J., Arafeh, R., Crowdis, J., Gang, S., Liu, D., AlDubayan, S. H.,
Salari, K., Kregel, S., Richter, C. et al. (2021), ‘Biologically informed deep neural network for
prostate cancer discovery’, Nature 598(7880), 348–352.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019), Optuna: A next-generation
hyperparameter optimization framework, in ‘Proceedings of the 25th ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data Mining’.
- LeCun, Y., Bottou, L., Orr, G. B. and Müller, K.-R. (2002), Efficient backprop, in ‘Neural
networks: Tricks of the trade’, Springer, pp. 9–50.
