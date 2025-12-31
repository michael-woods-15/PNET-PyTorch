from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, '_database')
PATHWAY_PATH = join(DATA_PATH, 'pathways')
REACTOME_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
DATA_SPLITS_PATH = join(PROSTATE_DATA_PATH, 'splits')
GENE_PATH = join(DATA_PATH, 'genes')
SELECTED_GENES_FILE_PATH = join(DATA_PATH, 'selected_genes.txt')