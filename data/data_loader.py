"""
Data loading module for CyTOF and scRNA-seq data classification.
Handles data loading, frequency calculation, and cell type mapping.
"""

import numpy as np
import pandas as pd
import scanpy as sc

# Cell type mapping dictionary for display
CELL_LABEL_MAP = {
    'Naive CD4+ T cells': 'Naive CD4+',
    'Central Memory and Effector Memory CD4+ T cells': 'CM/EM CD4+',
    'Effector Memory CD4+ T cells': 'EM CD4+',  # Added for validation dataset
    'Th1/Th2-like T cells': 'Th1/Th2',
    'Th17-like cells': 'Th17',
    'Regulatory T cells': 'Treg',
    'Terminal Effector CD4+ T cells': 'Term Effector CD4+',
    'Naive CD8+ T cells': 'Naive CD8+',
    'Central Memory and Effector Memory CD8+ T cells': 'CM/EM CD8+',
    'Central Memory and Terminal Effector CD8+ T cells': 'CM/Term CD8+',  # Added for validation dataset
    'Naive and Effector Memory CD8+ T cells': 'Naive/EM CD8+',  # Added for validation dataset
    'Terminal Effector CD8+ T cells': 'Term Effector CD8+',
    'TCRγδ T-cell': 'TCRγδ',
    'MAIT/NKT-like cells': 'MAIT/NKT',
    'Naive B cells': 'Naive B',
    'Memory B cells': 'Memory B',
    'Plasmablast B cells': 'Plasmablast B',
    'Transitional/activated B cells': 'Trans/Act B',
    'NK cell': 'NK',
    'Classical Monocytes': 'Classical Mono',
    'Transitional Monocytes': 'Transitional Mono',
    'Non-Classical Monocytes': 'Non-Classical Mono',
    'Regulatory Myeloid Cells/Monocytic Myeloid-derived Suppressor cells': 'Reg Myeloid',
    'Plasmacytoid Dendritic cells': 'Plasmacytoid DC',
    'Basophils': 'Basophils',
    'Neutrophils': 'Neutrophils',
    'Low density Neutrophils': 'Low density Neutrophils',
    'Eosinophils': 'Eosinophils',
    'Innate Lymphoid Cell Precursors': 'ILC Precursors',
    'Unknown': 'Unknown'  # Added for validation dataset
}

# Cell types to exclude from analysis
EXCLUDE_CELL_TYPES = [
    'Unknown',
    'artifacts',
    'Artifacts'  
]

# Cell types that are common between datasets - using their MAPPED (short) names
COMMON_CELL_TYPES_MAPPED = [
    'Basophils',
    'Classical Mono',
    'Eosinophils',
    'MAIT/NKT',
    'Memory B',
    'NK',
    'Naive B',
    'Naive CD4+',
    'Neutrophils',
    'Plasmablast B',
    'Plasmacytoid DC',
    'TCRγδ',
    'Term Effector CD4+',
    'Transitional Mono'
]

# Full cell type names that can be mapped to common types (for backward compatibility)
COMMON_CELL_TYPES = [
    'Basophils',
    'Classical Monocytes',
    'Eosinophils',
    'MAIT/NKT-like cells',
    'Memory B cells',
    'NK cell',
    'Naive B cells',
    'Naive CD4+ T cells',
    'Neutrophils',
    'Plasmablast B cells',
    'Plasmacytoid Dendritic cells',
    'TCRγδ T-cell',
    'Terminal Effector CD4+ T cells',
    'Transitional Monocytes'
]

# Labels to exclude from analysis
EXCLUDE_CELL_TYPES = [
    'Unknown',
    'artifacts'
]

def load_data(cytof_path, scrna_path):
    """
    Load CyTOF and scRNA-seq datasets.
    
    Args:
        cytof_path: Path to CyTOF AnnData h5ad file
        scrna_path: Path to scRNA-seq AnnData h5ad file
        
    Returns:
        Tuple of (cytof_data, scrna_data) as AnnData objects
    """
    print("Loading CyTOF data...")
    cytof_data = sc.read_h5ad(cytof_path)
    cytof_data.obs['Sample_ID'] = cytof_data.obs['Sample']
    
    print("Loading scRNA-seq data...")
    scrna_data = sc.read_h5ad(scrna_path)
    
    return cytof_data, scrna_data

def calculate_cytof_frequencies(adata, cell_type_col='cell_type'):
    """
    Calculate cell type frequencies for CyTOF data.
    
    Args:
        adata: AnnData object with CyTOF data
        cell_type_col: Column name in adata.obs containing cell type information
        
    Returns:
        DataFrame with cell type frequencies for each sample
    """
    print(f"Calculating frequencies using '{cell_type_col}' column")
    
    # Check for presence of required columns
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"'{cell_type_col}' column not found in data")
    
    # Get unique samples
    samples = adata.obs['Sample_ID'].unique()
    cell_types = adata.obs[cell_type_col].unique()
    
    # Remove artifacts if present
    if 'artifacts' in cell_types:
        cell_types = [ct for ct in cell_types if ct != 'artifacts']
        
    # Create empty dataframe to store frequencies
    freq_df = pd.DataFrame(index=samples, columns=cell_types)
    
    # Calculate frequencies
    for sample in samples:
        sample_cells = adata[adata.obs['Sample_ID'] == sample]
        
        # Exclude artifacts for total cell count
        valid_cells = sample_cells[sample_cells.obs[cell_type_col] != 'artifacts']
        total_cells = len(valid_cells)
        
        for cell_type in cell_types:
            ct_count = np.sum(sample_cells.obs[cell_type_col] == cell_type)
            freq_df.loc[sample, cell_type] = ct_count / total_cells if total_cells > 0 else 0
    
    # Add classifier information
    if 'Classifier' in adata.obs.columns:
        freq_df['Classifier'] = [adata[adata.obs['Sample_ID'] == sample].obs['Classifier'].iloc[0] 
                                for sample in samples]
    
    # Add ALSFRS-R scores if available
    alsfrs_cols = [col for col in adata.obs.columns if 'ALSFRS' in col]
    if alsfrs_cols:
        alsfrs_col = alsfrs_cols[0]
        freq_df['ALSFRS-R/time'] = [adata[adata.obs['Sample_ID'] == sample].obs[alsfrs_col].iloc[0] 
                                   for sample in samples]
    
    freq_df.columns = [CELL_LABEL_MAP.get(col, col) for col in freq_df.columns]
    
    print(f"Calculated frequencies for {len(samples)} samples and {len(cell_types)} cell types")
    return freq_df

def calculate_scrna_frequencies(adata, cell_type_col='cell_type'):
    """
    Calculate cell type frequencies for scRNA-seq data.
    
    Args:
        adata: AnnData object with scRNA-seq data
        cell_type_col: Column name in adata.obs containing cell type information
        
    Returns:
        DataFrame with cell type frequencies for each sample
    """
    print(f"Calculating scRNA-seq frequencies using '{cell_type_col}' column")
    
    # Check for presence of required columns
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"'{cell_type_col}' column not found in data")
        
    # Remove cells with excluded cell types
    if len(EXCLUDE_CELL_TYPES) > 0:
        excluded_mask = adata.obs[cell_type_col].isin(EXCLUDE_CELL_TYPES)
        if excluded_mask.sum() > 0:
            print(f"Removing {excluded_mask.sum()} cells with excluded cell types: {EXCLUDE_CELL_TYPES}")
            adata = adata[~excluded_mask].copy()
        else:
            print(f"No cells found with excluded cell types: {EXCLUDE_CELL_TYPES}")
    
    # Get unique samples
    samples = adata.obs['Sample_ID'].unique()
    cell_types = adata.obs[cell_type_col].unique()
    
    # Remove artifacts if present
    if 'artifacts' in cell_types:
        cell_types = [ct for ct in cell_types if ct != 'artifacts']
        
    # Create empty dataframe to store frequencies
    freq_df = pd.DataFrame(index=samples, columns=cell_types)
    
    # Calculate frequencies
    for sample in samples:
        sample_cells = adata[adata.obs['Sample_ID'] == sample]
        
        # Exclude artifacts for total cell count
        valid_cells = sample_cells[sample_cells.obs[cell_type_col] != 'artifacts']
        total_cells = len(valid_cells)
        
        for cell_type in cell_types:
            ct_count = np.sum(sample_cells.obs[cell_type_col] == cell_type)
            freq_df.loc[sample, cell_type] = ct_count / total_cells if total_cells > 0 else 0
    
    # Add classifier information
    if 'Classifier' in adata.obs.columns:
        freq_df['Classifier'] = [adata[adata.obs['Sample_ID'] == sample].obs['Classifier'].iloc[0] 
                                for sample in samples]
    
    # Add ALSFRS-R scores if available
    alsfrs_cols = [col for col in adata.obs.columns if 'ALSFRS' in col]
    if alsfrs_cols:
        alsfrs_col = alsfrs_cols[0]
        freq_df['ALSFRS-R/time'] = [adata[adata.obs['Sample_ID'] == sample].obs[alsfrs_col].iloc[0] 
                                   for sample in samples]
    
    print(f"Calculated frequencies for {len(samples)} samples and {len(cell_types)} cell types")
    return freq_df

def map_scrna_to_cytof_cell_types(scrna_freq):
    """
    Maps scRNA-seq cell types to CyTOF cell types based on defined mapping.
    
    Args:
        scrna_freq: DataFrame with scRNA-seq cell type frequencies
        
    Returns:
        DataFrame with CyTOF cell types as columns
    """
    print("Mapping scRNA-seq cell types to CyTOF cell types...")
    
    # Define mapping dictionary from scRNA-seq to CyTOF cell types
    # Format: 'CyTOF cell type': ['scRNA-seq cell type 1', 'scRNA-seq cell type 2', ...]
    mapping = {
        'Naive CD4+': ['Naive CD4+ T cells'],
        'CM/EM CD4+': ['Effector Memory CD4+ T cells', 'Central Memory CD4+/CD8+ T cells'],
        'Th1/Th2': ['Th2/Th17-like T cells'],  # Partial match
        'Th17': ['Th2/Th17-like T cells'],  # Partial match
        'Treg': ['Regulatory T cells'],
        'Term Effector CD4+': ['Terminal Effector CD4+ T cells'],
        'Naive CD8+': ['Naive CD8+ T cells'],
        'CM/EM CD8+': ['Effector Memory CD8+ T cells', 'Central Memory CD4+/CD8+ T cells'],
        'Term Effector CD8+': ['Terminal Effector CD8+ T cells', 'Exhausted CD8+ T cells'],
        'TCRγδ': ['TCRγδ T-cell'],
        'MAIT/NKT': ['MAIT cells', 'NKT-like cells'],
        'Naive B': ['Naive B cells'],
        'Memory B': ['Memory B cells'],
        'Plasmablast B': ['Plasmablast B cells'],
        'Trans/Act B': ['Transitional/activated B cells'],
        'NK': ['Cytotoxic NK cell', 'Memory NK cell'],
        'Classical Mono': ['Classical Monocytes'],
        'Transitional Mono': ['Transitional Monocytes'],
        'Non-Classical Mono': ['Non-Classical Monocytes', 'Macrophage', 'Exhausted myeloid cells'],
        'Reg Myeloid': ['Regulatory Myeloid Cells/Monocytic Myeloid-derived Suppressor cells'],
        'Plasmacytoid DC': ['Plasmacytoid Dendritic cells'],
        'Basophils': ['Basophils'],
        'Neutrophils': ['Neutrophils'],
        'Low density Neutrophils': ['Low density Neutrophils'],
        'Eosinophils': ['Eosinophils'],
        'ILC Precursors': ['Innate Lymphoid Cell Precursors']
    }
    
    # Find which scRNA-seq cell types are available in the data
    available_scrna_types = set(scrna_freq.columns) - {'Classifier', 'ALSFRS-R/time'}
    
    # Create mapping for classifier values from scRNA-seq to standardized naming
    classifier_mapping = {
        'Control': 'Healthy',          
        'Rapid': 'Rapid',              
        'Non-Rapid': 'Non-Rapid',      
        'Non-rapid': 'Non-Rapid',      
        'Rapid ALS': 'Rapid',          
        'Non-Rapid ALS': 'Non-Rapid',  
        'Non-rapid ALS': 'Non-Rapid',  
        'Fast': 'Rapid',               
        'Slow': 'Non-Rapid',           
        'Standard': 'Non-Rapid'        
    }
    
    # Initialize the mapped dataframe with the same index
    mapped_freq = pd.DataFrame(index=scrna_freq.index)
    
    # Map each CyTOF cell type from scRNA-seq frequencies
    mapped_cell_types = []
    for cytof_type, scrna_types in mapping.items():
        # Filter to available types in this dataset
        available_types = [t for t in scrna_types if t in available_scrna_types]
        
        if not available_types:  
            # Set to zero if no corresponding types in the data
            mapped_freq[cytof_type] = 0.0
            print(f"  No match found for {cytof_type}, setting to 0")
        else:
            # Sum frequencies from corresponding scRNA-seq cell types
            mapped_freq[cytof_type] = 0.0
            for scrna_type in available_types:
                mapped_freq[cytof_type] += scrna_freq[scrna_type].fillna(0)
            mapped_cell_types.append(cytof_type)
            type_list = ", ".join(available_types)
            print(f"  Mapped {cytof_type} from {type_list}")
    
    # Map classifier
    if 'Classifier' in scrna_freq.columns:
        mapped_freq['Classifier'] = scrna_freq['Classifier'].map(classifier_mapping)
        print(f"Mapped classifiers: {classifier_mapping}")
    
    # Add ALSFRS-R scores if they exist
    alsfrs_cols = [col for col in scrna_freq.columns if 'ALSFRS' in col]
    if alsfrs_cols:
        mapped_freq['ALSFRS-R/time'] = scrna_freq[alsfrs_cols[0]]
        print(f"Added ALSFRS scores from column {alsfrs_cols[0]}")
    
    print(f"Successfully mapped {len(mapped_cell_types)} cell types")
    return mapped_freq

def prepare_and_filter_data(cytof_data, scrna_data):
    """
    Prepare and filter CyTOF and scRNA-seq data to ensure compatibility for cross-platform validation.
    
    Args:
        cytof_data: AnnData object with CyTOF data
        scrna_data: AnnData object with scRNA-seq data
        
    Returns:
        Tuple of (cytof_freq_filtered, mapped_scrna_freq_filtered) as DataFrames
    """
    print("\nPreparing and filtering data for cross-platform validation...")
    
    cytof_freq = calculate_cytof_frequencies(cytof_data)
    scrna_freq = calculate_scrna_frequencies(scrna_data)
    
    # Map scRNA-seq cell types to CyTOF cell types
    mapped_scrna_freq = map_scrna_to_cytof_cell_types(scrna_freq)
    
    # Identify cell types available in both datasets before creating derived features
    cytof_basic = [col for col in cytof_freq.columns if col not in ['Classifier', 'ALSFRS-R/time']]
    scrna_basic = [col for col in mapped_scrna_freq.columns if col not in ['Classifier', 'ALSFRS-R/time']]
    
    # Check which cell types have non-zero values in each dataset
    zero_in_scrna = []
    for col in scrna_basic:
        if (mapped_scrna_freq[col] == 0).all():
            zero_in_scrna.append(col)
    
    # Remove zero-valued columns from consideration
    scrna_nonzero = [col for col in scrna_basic if col not in zero_in_scrna]
    print(f"\nFound {len(zero_in_scrna)} cell types that are all zeros in scRNA-seq:")
    for ct in sorted(zero_in_scrna):
        print(f"  - {ct}")
    
    # Find truly common cell types between CyTOF and scRNA-seq
    common_basic_celltypes = set(cytof_basic) & set(scrna_nonzero)
    
    # Print information about the common cell types with actual values
    print(f"\nFound {len(common_basic_celltypes)} cell types with actual values common to both datasets:")
    for ct in sorted(common_basic_celltypes):
        print(f"  - {ct}")
    
    # missing cell types
    missing_celltypes_cytof = set(cytof_basic) - common_basic_celltypes
    missing_celltypes_scrna = set(scrna_nonzero) - common_basic_celltypes
    
    if missing_celltypes_cytof:
        print(f"\nWarning: The following {len(missing_celltypes_cytof)} cell types are in CyTOF but missing in scRNA-seq:")
        for ct in sorted(missing_celltypes_cytof):
            print(f"  - {ct}")
        print("These cell types will be excluded from all feature sets")
    
    if missing_celltypes_scrna:
        print(f"\nNote: The following {len(missing_celltypes_scrna)} cell types are in scRNA-seq but not in CyTOF:")
        for ct in sorted(missing_celltypes_scrna):
            print(f"  - {ct}")
    
    # Create filtered copies of both datasets with only common cell types + metadata
    cytof_freq_filtered = cytof_freq.copy()
    mapped_scrna_freq_filtered = mapped_scrna_freq.copy()
    
    # Identify cell types that are all zeros in either dataset
    zero_in_cytof = []
    for col in cytof_basic:
        if col in cytof_freq_filtered.columns and (cytof_freq_filtered[col] == 0).all():
            zero_in_cytof.append(col)
    
    # Define problematic cell types to exclude (either not common or all-zero)
    cytof_to_remove = set([col for col in cytof_basic if col not in common_basic_celltypes]) | set(zero_in_cytof)
    scrna_to_remove = set([col for col in scrna_basic if col not in common_basic_celltypes]) | set(zero_in_scrna)
    
    # Remove problematic cell types from CyTOF data
    for col in cytof_to_remove:
        if col in cytof_freq_filtered.columns:
            cytof_freq_filtered.drop(columns=[col], inplace=True)
    
    # Remove problematic cell types from scRNA-seq data
    for col in scrna_to_remove:
        if col in mapped_scrna_freq_filtered.columns:
            mapped_scrna_freq_filtered.drop(columns=[col], inplace=True)
            
    # Print what was removed due to being all-zeros
    if zero_in_cytof:
        print(f"\nRemoved {len(zero_in_cytof)} all-zero cell types from CyTOF data:")
        for col in sorted(zero_in_cytof):
            print(f"  - {col}")
            
    if zero_in_scrna:
        print(f"\nRemoved {len(zero_in_scrna)} all-zero cell types from scRNA-seq data:")
        for col in sorted(zero_in_scrna):
            print(f"  - {col}")
    
    return cytof_freq_filtered, mapped_scrna_freq_filtered

def prepare_data_for_task(cytof_freq, scrna_freq, common_features, task):
    """
    Prepare data for a specific classification task with balanced class distribution.
    
    Args:
        cytof_freq: CyTOF frequency data
        scrna_freq: scRNA-seq frequency data
        common_features: List of features to use
        task: Dictionary with task information (task_id, task_name, label_map)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, class_names)
    """
    task_id, task_name, label_map = task
    print(f"\nPreparing data for task: {task_name}")
    
    # Standardize CyTOF classifier names
    cytof_classifier = cytof_freq['Classifier'].copy()
    cytof_classifier = cytof_classifier.replace({
        'Healthy': 'Healthy',     # keep as is
        'Fast': 'Rapid',          # rename Fast to Rapid
        'Slow': 'Non-Rapid',      # rename Slow to Non-Rapid
        'Standard': 'Non-Rapid'   # rename Standard to Non-Rapid
    })
    
    # Prepare CyTOF training data
    y_cytof = cytof_classifier.map(label_map)
    filtered_indices = y_cytof[y_cytof.notna()].index
    X_train = cytof_freq.loc[filtered_indices, common_features]
    y_train = y_cytof.loc[filtered_indices]
    
    # Prepare scRNA-seq validation data
    y_scrna = scrna_freq['Classifier'].map(label_map)
    filtered_indices = y_scrna[y_scrna.notna()].index
    X_val = scrna_freq.loc[filtered_indices, common_features]
    y_val = y_scrna.loc[filtered_indices]
    
    # Get class names
    class_names = sorted(np.unique(pd.concat([y_train, y_val])))
    
    # Print class distributions
    print(f"Training set (CyTOF): {len(X_train)} samples")
    print(f"  Class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Validation set (scRNA-seq): {len(X_val)} samples")
    print(f"  Class distribution: {pd.Series(y_val).value_counts().to_dict()}")
    
    return X_train, y_train, X_val, y_val, class_names

def create_balanced_bootstrap(X, y):
    """
    Create balanced dataset using bootstrap downsampling.
    
    Args:
        X: Features DataFrame or array
        y: Target labels
        
    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    print("Creating balanced bootstrap sample...")
    
    # Get class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    print(f"Class counts: {dict(zip(unique_classes, class_counts))}")
    print(f"Using min class count: {min_class_count}")
    
    # Initialize arrays for balanced data
    X_balanced = []
    y_balanced = []
    
    # For each class, either downsample (majority) or bootstrap (minority)
    for class_label in unique_classes:
        # Find indices of this class
        class_indices = np.where(y == class_label)[0]
        
        # If this is the majority class, downsample
        if len(class_indices) > min_class_count:
            # Randomly select min_class_count samples without replacement
            sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=False)
            print(f"  Downsampled class {class_label}: {len(class_indices)} → {min_class_count}")
        else:
            # For minority classes, bootstrap with replacement to reach min_class_count
            sampled_indices = np.random.choice(class_indices, size=min_class_count, replace=True)
            print(f"  Bootstrapped class {class_label}: {len(class_indices)} → {min_class_count}")
        
        # Add selected samples to balanced dataset
        if isinstance(X, pd.DataFrame):
            X_class_sampled = X.iloc[sampled_indices]
            y_class_sampled = y.iloc[sampled_indices] if isinstance(y, pd.Series) else y[sampled_indices]
        else:
            X_class_sampled = X[sampled_indices]
            y_class_sampled = y[sampled_indices]
        
        X_balanced.append(X_class_sampled)
        y_balanced.append(y_class_sampled)
    
    # Combine balanced samples for all classes
    if isinstance(X, pd.DataFrame):
        X_balanced = pd.concat(X_balanced)
        y_balanced = pd.concat(y_balanced) if isinstance(y, pd.Series) else np.concatenate(y_balanced)
    else:
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)
    
    print(f"Created balanced dataset with {len(X_balanced)} samples")
    print(f"  New class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    return X_balanced, y_balanced