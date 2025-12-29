import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')

def save_genes_list(genes_list, output_path):
    logging.info(f"Saving {len(genes_list)} genes to {output_path}")
    with open(output_path, 'w') as f:
        for gene in genes_list:
            f.write(f"{gene}\n")
    logging.info(f"Genes list saved successfully")


def load_genes_list(input_path):
    logging.info(f"Loading genes list from {input_path}")
    with open(input_path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    logging.info(f"Loaded {len(genes)} genes")
    return genes

