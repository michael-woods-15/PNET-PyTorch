from os.path import join, realpath, dirname

BASE_PATH = dirname(realpath(__file__))
DATA_PATH = join(BASE_PATH, '_database')
PATHWAY_PATH = join(DATA_PATH, 'pathways')
REACTOME_PATHWAY_PATH = join(PATHWAY_PATH, 'Reactome')
PROSTATE_DATA_PATH = join(DATA_PATH, 'prostate')
GENE_PATH = join(DATA_PATH, 'genes')