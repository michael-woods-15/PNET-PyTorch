import os
import pandas as pd
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from config_path import REACTOME_PATHWAY_PATH

REACTOME_PATHWAYS_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.gmt')
HIERARCHIES_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathwaysRelation.txt')
PATHWAYS_NAMES_FILE = os.path.join(REACTOME_PATHWAY_PATH, 'ReactomePathways.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')


def load_reactome_pathways_file(filename, genes_col, pathway_col):
    logging.info("Loading reactome pathways from .gmt file")
    pathway_dict_list = []

    with open(filename) as gmt:
        file_contents = gmt.readlines()

        for row in file_contents:
            genes = row.strip().split('\t')

            for gene in genes[genes_col : ]:
                pathway = genes[pathway_col]
                pathway_dict = {'group': pathway, 'gene': gene}
                pathway_dict_list.append(pathway_dict)
                
    reactome_pathways_df = pd.DataFrame(pathway_dict_list)

    logging.info(f"Size of Reactome Pathways df: {reactome_pathways_df.size}")

    return reactome_pathways_df


def load_reactome_hierarchies(filename):
    df = pd.read_csv(filename, sep='\t')
    df.columns = ['child', 'parent']
    return df


def load_pathway_names(filename):
    df = pd.read_csv(filename, sep='\t')
    df.columns = ['reactome_id', 'pathway_name', 'species']
    return df


if __name__ == "__main__":
    reactome_pathways = load_reactome_pathways_file(filename=REACTOME_PATHWAYS_FILE, genes_col=3, pathway_col=1)
    reactome_hierarchies = load_reactome_hierarchies(HIERARCHIES_FILE)
    reactome_pathway_names = load_pathway_names(PATHWAYS_NAMES_FILE)
