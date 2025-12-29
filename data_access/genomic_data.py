import logging
import numpy as np
import pandas as pd
import sys
import os  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from config_path import PROSTATE_DATA_PATH, GENE_PATH, SELECTED_GENES_FILE_PATH
from data_utils import save_genes_list

# Paths
PROCESSED_PATH = os.path.join(PROSTATE_DATA_PATH, 'processed')
SPLITS_PATH = os.path.join(PROSTATE_DATA_PATH, 'splits')
SELECTED_GENES_PATH = os.path.join(GENE_PATH, 'tcga_prostate_expressed_genes_and_cancer_genes.csv')

# Data files
MUTATION_FILE = 'P1000_final_analysis_set_cross_important_only.csv'
CNV_FILE = 'P1000_data_CNA_paper.csv'
RESPONSES_FILE = 'response_paper.csv'


def load_data(filename):
    filepath = os.path.join(PROCESSED_PATH, filename)
    logging.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, index_col=0)
    logging.info(f"Loaded dataframe with shape {data.shape}")
    return data


def load_response_labels():
    logging.info("Loading response labels")
    labels = pd.read_csv(os.path.join(PROCESSED_PATH, RESPONSES_FILE))
    labels = labels.set_index('id')
    logging.info(f"Loaded {labels.shape[0]} response labels")
    return labels


def load_selected_genes():
    logging.info(f"Loading selected genes from {SELECTED_GENES_PATH}")
    selected_genes_df = pd.read_csv(SELECTED_GENES_PATH, header=0)
    selected_genes = selected_genes_df['genes'].tolist()
    logging.info(f"Loaded {len(selected_genes)} selected genes")
    return selected_genes


def process_mutation_data(df):
    """
    Process mutation data: cap values at 1.
    
    Args:
        df: Mutation dataframe
    
    Returns:
        Processed mutation dataframe
    """
    logging.info("Processing mutation data")
    df = df.copy()
    df[df > 1.0] = 1.0
    return df


def process_cnv_data(df):
    """
    Process CNV data: split into amplifications and deletions.
    
    Args:
        df: CNV dataframe with values in {-2, -1, 0, 1, 2}
    
    Returns:
        Tuple of (amplification_df, deletion_df)
    """
    logging.info("Processing CNV data")
    
    # Amplifications: 2 -> 1, everything else -> 0
    amp_df = df.copy(deep=True)
    amp_df[amp_df <= 0.0] = 0.0
    amp_df[amp_df == 1.0] = 0.0
    amp_df[amp_df == 2.0] = 1.0
    
    # Deletions: -2 -> 1, everything else -> 0
    del_df = df.copy(deep=True)
    del_df[del_df >= 0.0] = 0.0
    del_df[del_df == -1.0] = 0.0
    del_df[del_df == -2.0] = 1.0
    
    logging.info(f"Created amplification data: {amp_df.shape}")
    logging.info(f"Created deletion data: {del_df.shape}")
    
    return amp_df, del_df


def reindex_to_selected_genes(df, selected_genes, data_type_name):
    """
    Reindex dataframe to include all selected genes, filling missing with 0.
    
    Args:
        df: Input dataframe with genes as columns
        selected_genes: List of genes to include
        data_type_name: Name of data type (for logging)
    
    Returns:
        Reindexed dataframe
    """
    original_genes = set(df.columns)
    selected_genes_set = set(selected_genes)
    
    missing_genes = selected_genes_set - original_genes
    extra_genes = original_genes - selected_genes_set
    
    if missing_genes:
        logging.info(f"{data_type_name}: {len(missing_genes)} genes from selected list not in data (will be filled with 0)")
    if extra_genes:
        logging.info(f"{data_type_name}: {len(extra_genes)} genes in data not in selected list (will be dropped)")
    
    # Reindex to selected genes
    df_reindexed = df.reindex(columns=selected_genes, fill_value=0)
    logging.info(f"{data_type_name} after reindexing: {df_reindexed.shape}")
    
    return df_reindexed


def combine_modalities(df_list, data_type_names):
    """
    Combine multiple genomic dataframes with multi-level column indexing.
    
    Args:
        df_list: List of dataframes (all must have same columns/genes)
        data_type_names: List of names for each data type
    
    Returns:
        Combined dataframe with multi-level columns (gene, data_type)
    """
    logging.info("Combining data modalities with multi-level indexing")
    
    # Verify all dataframes have the same columns
    first_cols = set(df_list[0].columns)
    for i, df in enumerate(df_list[1:], 1):
        if set(df.columns) != first_cols:
            raise ValueError(f"Dataframe {i} has different columns than dataframe 0")
    
    # Concatenate with multi-level columns: (data_type, gene)
    combined_df = pd.concat(df_list, keys=data_type_names, join='inner', axis=1)
    
    # Swap levels to (gene, data_type)
    combined_df = combined_df.swaplevel(i=0, j=1, axis=1)
    
    combined_df = combined_df.sort_index(axis=1, level=0)
    
    logging.info(f"Combined dataframe shape: {combined_df.shape}")
    logging.info(f"Number of samples: {combined_df.shape[0]}")
    logging.info(f"Number of unique genes: {len(combined_df.columns.get_level_values(0).unique())}")
    logging.info(f"Number of data types per gene: {len(data_type_names)}")
    
    return combined_df


def join_with_labels_and_clean(genomic_df, labels_df):
    """
    Join genomic data with response labels and remove samples with missing responses.
    
    Args:
        genomic_df: Genomic dataframe with multi-level columns (gene, data_type)
        labels_df: Labels dataframe with 'response' column
    
    Returns:
        Tuple of (features_df, response_series, sample_ids)
    """
    logging.info("Joining genomic data with response labels")
    
    common_samples = genomic_df.index.intersection(labels_df.index)
    logging.info(f"Samples in genomic data: {len(genomic_df.index)}")
    logging.info(f"Samples in labels: {len(labels_df.index)}")
    logging.info(f"Common samples (inner join): {len(common_samples)}")
    
    genomic_filtered = genomic_df.loc[common_samples]
    labels_filtered = labels_df.loc[common_samples]
    
    non_null_mask = ~labels_filtered['response'].isnull()
    samples_before = len(labels_filtered)
    samples_after = non_null_mask.sum()
    
    if samples_before > samples_after:
        logging.warning(f"Removed {samples_before - samples_after} samples with null responses")
    
    features = genomic_filtered.loc[non_null_mask]
    response = labels_filtered.loc[non_null_mask, 'response']
    samples = features.index
    
    logging.info(f"Final dataset shape: {features.shape}")
    logging.info(f"Final number of samples: {len(samples)}")
    
    return features, response, samples


def get_genes_list(df_list, data_type_names, use_selected_genes_only, use_coding_genes_only, combine_type):
    """
    Get the union/intersection of all genes across all modalities.
    
    Args:
        df_list: List of dataframes
        data_type_names: Names of data types (for logging)
        use_selected_genes_only: Boolean value to filter gene list
        use_coding_genes_only: Boolean value to further filter selected genes
        combine_type: determines union or intersection of genes
    
    Returns:
        List of all unique genes or intersection of genes across all modalities
    """
    logging.info("Computing union of genes across all modalities")
    
    gene_sets = []
    for df, name in zip(df_list, data_type_names):
        genes = set(df.columns)
        logging.info(f"{name}: {len(genes)} genes")
        gene_sets.append(genes)
    
    if combine_type == 'union':
        all_genes = set.union(*gene_sets)
    elif combine_type == 'intersection':
        all_genes = set.intersection(*gene_sets)
    else:
        raise ValueError("combine_type must be either union or intersection")
    
    logging.info(f"Total unique genes across all modalities: {len(all_genes)}")
    
    for name, gene_set in zip(data_type_names, gene_sets):
        missing = len(all_genes) - len(gene_set)
        logging.info(f"{name}: missing {missing} genes that exist in other modalities (will be filled with 0)")

    if use_selected_genes_only:
        selected_genes = load_selected_genes()

        selected_genes_set = set(selected_genes)
        genes_before = len(all_genes)
        all_genes = all_genes.intersection(selected_genes_set)
        
        logging.info(f"Filtered to selected genes: {genes_before} -> {len(all_genes)}")
        
        missing_from_data = selected_genes_set - set(all_genes)
        if missing_from_data:
            logging.warning(f"{len(missing_from_data)} selected genes don't exist in the data (after previous filters)")
        

    if use_coding_genes_only:
        f = os.path.join(GENE_PATH, "HUGO_genes/protein-coding_gene_with_coordinate_minimal.txt")
        coding_genes_df = pd.read_csv(f, sep='\t', header=None)
        logging.info(f'Coding Genes Shape {coding_genes_df.shape}')
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())
        all_genes = all_genes.intersection(coding_genes)

    
    all_genes = sorted(list(all_genes))
    
    return all_genes


def main(use_selected_genes_only, use_coding_genes_only, combine_type):
    """Main data processing pipeline."""
    
    logging.info("="*80)
    logging.info("Starting Prostate Cancer Data Processing Pipeline")
    logging.info("="*80)
    
    # Step 1: Load response labels
    labels = load_response_labels()
    
    # Step 2: Load and process mutation data
    logging.info("\n" + "="*80)
    logging.info("Processing Mutation Data")
    logging.info("="*80)
    mut_df = load_data(MUTATION_FILE)
    mut_processed = process_mutation_data(mut_df)
    
    # Step 3: Load and process CNV data
    logging.info("\n" + "="*80)
    logging.info("Processing CNV Data")
    logging.info("="*80)
    cnv_df = load_data(CNV_FILE)
    cnv_amp, cnv_del = process_cnv_data(cnv_df)
    
    # Step 4: Get union of all genes across modalities
    logging.info("\n" + "="*80)
    logging.info("Finding Union of Genes Across Modalities")
    logging.info("="*80)
    genomic_dfs_raw = [mut_processed, cnv_amp, cnv_del]
    data_type_names = ['mutation', 'cnv_amp', 'cnv_del']
    all_genes = get_genes_list(genomic_dfs_raw, data_type_names, use_selected_genes_only, use_coding_genes_only, combine_type)
    save_genes_list(all_genes, SELECTED_GENES_FILE_PATH)
    
    # Step 5: Reindex each modality to have ALL genes (union)
    logging.info("\n" + "="*80)
    logging.info("Reindexing All Modalities to Union of Genes")
    logging.info("="*80)
    mut_reindexed = reindex_to_selected_genes(mut_processed, all_genes, "Mutations")
    cnv_amp_reindexed = reindex_to_selected_genes(cnv_amp, all_genes, "CNV Amplifications")
    cnv_del_reindexed = reindex_to_selected_genes(cnv_del, all_genes, "CNV Deletions")
    
    # Step 6: Combine all modalities with multi-level indexing
    logging.info("\n" + "="*80)
    logging.info("Combining Data Modalities")
    logging.info("="*80)
    genomic_dfs = [mut_reindexed, cnv_amp_reindexed, cnv_del_reindexed]
    merged_df = combine_modalities(genomic_dfs, data_type_names)
    
    # Step 7: Join with labels and clean
    logging.info("\n" + "="*80)
    logging.info("Joining with Labels and Final Cleanup")
    logging.info("="*80)
    x, y, samples = join_with_labels_and_clean(merged_df, labels)
    
    # Step 8: Summary
    logging.info("\n" + "="*80)
    logging.info("Pipeline Complete - Final Dataset Summary")
    logging.info("="*80)
    logging.info(f"Number of samples: {x.shape[0]}")
    logging.info(f"Number of features (genes × data_types): {x.shape[1]}")
    logging.info(f"Number of unique genes: {len(x.columns.get_level_values(0).unique())}")
    logging.info(f"Number of responses: {y.shape[0]}")
    logging.info(f"Response distribution:\n{y.value_counts()}")
    
    # Verify integrity
    assert x.shape[0] == y.shape[0] == len(samples), "Mismatch in sample counts"
    assert x.isnull().sum().sum() == 0, "Found unexpected null values in features"
    assert y.isnull().sum() == 0, "Found unexpected null values in response"
    
    logging.info("\nData integrity checks passed")
    
    return x, y, samples


if __name__ == "__main__":
    features, response, sample_ids = main(use_selected_genes_only = True, use_coding_genes_only = True, combine_type = 'union')