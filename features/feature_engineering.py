"""
Feature engineering module for CyTOF and scRNA-seq data.
Creates derived features including log-transformations, interactions, and ratios.
See feature_sets_documentation.csv for complete feature set details.
"""

import os
import numpy as np
import pandas as pd

# Cell label mapping for shorter names 
CELL_LABEL_MAP = {
    'Naive CD4+ T cells': 'Naive CD4+',
    'Central Memory and Effector Memory CD4+ T cells': 'CM/EM CD4+',
    'Th1/Th2-like T cells': 'Th1/Th2',
    'Th17-like cells': 'Th17',
    'Regulatory T cells': 'Treg',
    'Terminal Effector CD4+ T cells': 'Term Effector CD4+',
    'Naive CD8+ T cells': 'Naive CD8+',
    'Central Memory and Effector Memory CD8+ T cells': 'CM/EM CD8+',
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
    'Innate Lymphoid Cell Precursors': 'ILC Precursors'
}

THRESHOLDS = {
    'Slow': {'p_value': 0.05, 'fdr': 0.001, 'correlation': 0.5, 'variance': 0.38, 'cv': 3},
    'Fast': {'p_value': 0.05, 'fdr': 0.001, 'correlation': 0.5, 'variance': 0.38, 'cv': 3},
    'Healthy': {'p_value': 0.05, 'fdr': 0.001, 'correlation': 0.5, 'variance': 0.38, 'cv': 3},
    'Standard': {'p_value': 0.05, 'fdr': 0.001, 'correlation': 0.5, 'variance': 0.38, 'cv': 3}
}

def extract_significant_correlations(csv_file=None):
    """
    Extract significant correlations from the correlation results CSV
    using the same thresholds as in the correlation analysis
    
    Args:
        csv_file: Path to correlation results CSV. If None, uses 'correlation_results.csv' in the project root.
        
    Returns:
        Dictionary of significant correlations by classifier
    """
    # Default path if not provided
    if csv_file is None:
        # Simple relative path - users can modify this if needed
        csv_file = 'correlation_results.csv'
    
    print(f"Looking for correlation file at: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"Warning: Could not find the correlation results file at {csv_file}")
        print("Using default hardcoded correlations instead")
        return get_default_correlations()
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded correlation data from {csv_file} with {len(df)} rows")
    
    # Dictionary to store significant correlations by classifier
    sig_corrs = {
        'Healthy': [],
        'Slow': [],
        'Standard': [],
        'Fast': []
    }
    
    # Process each classifier
    for classifier in sig_corrs.keys():
        # Apply thresholds
        thresh = THRESHOLDS[classifier]
        df_class = df[df['Classifier'] == classifier].copy()
        
        filtered = df_class[
            (df_class['FDR_corrected_pvalue'] <= thresh['fdr']) &
            (df_class['P_value'] <= thresh['p_value']) &
            (df_class['CV_Cell_Type_1'] <= thresh['cv']) &
            (df_class['CV_Cell_Type_2'] <= thresh['cv']) &
            (df_class['Correlation'].abs() >= thresh['correlation']) &
            (df_class['Correlation_Variance'] <= thresh['variance'])
        ]
        
        # Convert to short names and store as tuples
        for _, row in filtered.iterrows():
            cell1 = row['Cell_Type_1']
            cell2 = row['Cell_Type_2']
            
            # Skip self-correlations
            if cell1 == cell2:
                continue
            
            # Use short names
            short_cell1 = CELL_LABEL_MAP.get(cell1, cell1)
            short_cell2 = CELL_LABEL_MAP.get(cell2, cell2)
            
            # Store as tuple (alphabetically sorted to avoid duplicates)
            pair = tuple(sorted([short_cell1, short_cell2]))
            if pair not in sig_corrs[classifier]:
                sig_corrs[classifier].append(pair)
        
        print(f"Found {len(sig_corrs[classifier])} significant correlations for {classifier}")
    
    return sig_corrs

def get_default_correlations():
    """Return default correlations if CSV file can't be loaded"""
    correlations_by_group = {
        'Slow': [
            ('Basophils', 'Transitional Mono'),
            ('CM/EM CD4+', 'Classical Mono'),
            ('CM/EM CD4+', 'Naive CD4+'),
            ('CM/EM CD8+', 'Term Effector CD8+'),
            ('CM/EM CD8+', 'Th1/Th2'),
            ('CM/EM CD8+', 'Transitional Mono'),
            ('Classical Mono', 'Naive B'),
            ('Naive B', 'Plasmablast B'),
            ('Naive B', 'Th1/Th2'),
            ('Naive CD4+', 'Plasmacytoid DC'),
            ('Naive CD4+', 'Plasmablast B'),
            ('Naive CD4+', 'Th1/Th2'),
            ('Non-Classical Mono', 'Transitional Mono'),
            ('Plasmablast B', 'Plasmacytoid DC'),
            ('Plasmablast B', 'Th1/Th2'),
            ('Term Effector CD4+', 'Term Effector CD8+')
        ],
        'Fast': [
            ('CM/EM CD4+', 'Naive CD4+'),
            ('Classical Mono', 'Non-Classical Mono'),
            ('Eosinophils', 'Plasmacytoid DC'),
            ('ILC Precursors', 'Reg Myeloid'),
            ('NK', 'Th1/Th2'),
            ('Naive B', 'Treg'),
            ('Naive CD4+', 'Neutrophils'),
            ('Term Effector CD4+', 'Term Effector CD8+')
        ],
        'Standard': [
            ('CM/EM CD4+', 'Classical Mono'),
            ('CM/EM CD8+', 'Neutrophils'),
            ('CM/EM CD8+', 'Reg Myeloid'),
            ('CM/EM CD8+', 'Th1/Th2'),
            ('Classical Mono', 'Low density Neutrophils'),
            ('Eosinophils', 'Naive B'),
            ('Eosinophils', 'Term Effector CD4+'),
            ('ILC Precursors', 'Reg Myeloid'),
            ('ILC Precursors', 'Treg'),
            ('ILC Precursors', 'Th1/Th2'),
            ('NK', 'Term Effector CD4+'),
            ('Naive CD4+', 'Plasmablast B'),
            ('Naive CD8+', 'Plasmablast B'),
            ('Neutrophils', 'Th1/Th2'),
            ('Reg Myeloid', 'Treg'),
            ('Reg Myeloid', 'Th1/Th2'),
            ('Term Effector CD4+', 'Term Effector CD8+')
        ],
        'Healthy': []
    }
    
    return correlations_by_group

def extract_task_specific_correlations(csv_file=None, task=None):
    """
    Extract correlations specifically relevant to a given classification task.
    
    Args:
        csv_file: Path to correlation results CSV. If None, uses 'correlation_results.csv' in the project root.
        task: Task name as string (e.g., 'Healthy vs ALS', 'Fast vs Slow')
        
    Returns:
        List of significant correlation pairs specific to the task
    """
    # Default path if not provided
    if csv_file is None:
        # Simple relative path - users can modify this if needed
        csv_file = 'correlation_results.csv'
    
    # Parse the task to determine which classifiers to include
    task_classifiers = []
    
    if task is None:
        # If no task is specified, return empty list
        return []
        
    if 'Healthy' in task:
        task_classifiers.append('Healthy')
    
    if 'Fast' in task or 'Rapid' in task:
        task_classifiers.append('Fast')
        
    if 'Slow' in task or 'Non-Rapid' in task:
        task_classifiers.append('Slow')
        
    if 'Standard' in task:
        task_classifiers.append('Standard')
        
    # If "ALS" is used, include all disease states
    if 'ALS' in task and 'Healthy' in task:
        if 'Fast' not in task_classifiers:
            task_classifiers.append('Fast')
        if 'Slow' not in task_classifiers:
            task_classifiers.append('Slow')
        if 'Standard' not in task_classifiers:
            task_classifiers.append('Standard')
            
    print(f"Extracting correlations for task '{task}' using classifiers: {task_classifiers}")
    
    if not os.path.exists(csv_file) or not task_classifiers:
        print(f"Warning: Could not find the correlation file or no valid classifiers for task")
        return []
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded correlation data from {csv_file} with {len(df)} rows")
    
    # List to store task-specific correlations
    task_correlations = []
    
    # Process each classifier relevant to the task
    for classifier in task_classifiers:
        # Apply thresholds
        thresh = THRESHOLDS[classifier]
        df_class = df[df['Classifier'] == classifier].copy()
        
        filtered = df_class[
            (df_class['FDR_corrected_pvalue'] <= thresh['fdr']) &
            (df_class['P_value'] <= thresh['p_value']) &
            (df_class['CV_Cell_Type_1'] <= thresh['cv']) &
            (df_class['CV_Cell_Type_2'] <= thresh['cv']) &
            (df_class['Correlation'].abs() >= thresh['correlation']) &
            (df_class['Correlation_Variance'] <= thresh['variance'])
        ]
        
        # Convert to short names and store as tuples
        for _, row in filtered.iterrows():
            cell1 = row['Cell_Type_1']
            cell2 = row['Cell_Type_2']
            
            # Skip self-correlations
            if cell1 == cell2:
                continue
            
            # Use short names
            short_cell1 = CELL_LABEL_MAP.get(cell1, cell1)
            short_cell2 = CELL_LABEL_MAP.get(cell2, cell2)
            
            # Store as tuple (alphabetically sorted to avoid duplicates)
            pair = tuple(sorted([short_cell1, short_cell2]))
            if pair not in task_correlations:
                task_correlations.append(pair)
        
    print(f"Found {len(task_correlations)} task-specific correlations for {task}")
    return task_correlations

def create_feature_sets(freq_df, filter_lists=True, task_name=None):
    """
    Create different feature sets for modeling.
    
    This function creates several categories of features:
    1. Base features: Log-transformed cell frequencies
    2. Derived interaction features: Products of cell type frequencies
    3. Derived ratio features: Log ratios between cell type frequencies
       - From significant correlations 
       - Between significant cell types
       - Between all cell type pairs
    4. Task-specific features: Features derived from correlations relevant to specific tasks
    
    Feature set naming conventions:
    - correlation_*: Features derived from significant correlations
    - significant_celltype_*: Features involving only significant cell types
    - all_celltype_*: Features involving all cell types
    - *_with_*: Combined feature sets (base features + derived features)
    
    Args:
        freq_df: DataFrame with frequency data
        filter_lists: If True, filter lists to only include cell types in the data
        task_name: Name of the classification task (e.g., 'Healthy vs Fast')

    Returns:
        Tuple of (freq_df with derived features, dict of feature sets)
    """
    print("Creating feature sets…")

    # 1) Extract significant correlation pairs
    sig_corrs_dict = extract_significant_correlations()
    original_significant_correlations = []
    for lst in sig_corrs_dict.values():
        original_significant_correlations.extend(lst)
    original_significant_correlations = list(set(original_significant_correlations))
    print(f"Found {len(original_significant_correlations)} total significant correlation pairs")

    # 2) Filter pairs whose columns aren't present
    if filter_lists:
        significant_correlations = [
            (c1, c2)
            for c1, c2 in original_significant_correlations
            if c1 in freq_df.columns and c2 in freq_df.columns
        ]
        removed = len(original_significant_correlations) - len(significant_correlations)
        if removed:
            print(f"Filtered out {removed} correlation pairs missing from data")
    else:
        significant_correlations = original_significant_correlations
        
    # Get list of all cell types (excluding metadata columns)
    all_cell_types = [
        col for col in freq_df.columns 
        if col not in ['Classifier', 'ALSFRS-R/time', 'artifacts', 'Artifacts', 'Unknown']
    ]
    
    # Define significant cell types based on research findings
    original_sig_cell_types = [
        'ILC Precursors',
        'Plasmacytoid DC',
        'Transitional Mono',
        'Reg Myeloid',
        'TCRγδ',
        'Treg'
    ]
    
    # Filter significant cell types if needed
    sig_cell_types = [ct for ct in original_sig_cell_types if ct in all_cell_types]
    
    if filter_lists and len(sig_cell_types) < len(original_sig_cell_types):
        missing = set(original_sig_cell_types) - set(sig_cell_types)
        print(f"Warning: Filtered out {len(missing)} significant cell types not present in this dataset:")
        for ct in missing:
            print(f"  - {ct}")

    task_specific_correlations = []
    if task_name:
        task_specific_correlations = extract_task_specific_correlations(task=task_name)
        task_specific_correlations = [
            (c1, c2)
            for c1, c2 in task_specific_correlations
            if c1 in freq_df.columns and c2 in freq_df.columns
        ]
        print(f"Extracted {len(task_specific_correlations)} task-specific correlation pairs for '{task_name}'")

    correlation_interaction_features = []
    sig_interaction_features = []
    correlation_ratio_features = []
    sig_only_ratio_features = []
    all_log_cell_types = []
    for ct in all_cell_types:
        try:
            arr = np.array(freq_df[ct].values, dtype=float)
            col = f"{ct}_log"
            freq_df[col] = np.log1p(arr)
            all_log_cell_types.append(col)
        except Exception as e:
            print(f"Error log-transforming {ct}: {str(e)}")
            continue

    sig_log_cell_types = [f"{ct}_log" for ct in sig_cell_types if f"{ct}_log" in all_log_cell_types]
    for ct1, ct2 in significant_correlations:
        try:
            col_name = f"{ct1}_{ct2}_interaction"
            arr1 = np.array(freq_df[ct1].values, dtype=float)
            arr2 = np.array(freq_df[ct2].values, dtype=float)
            freq_df[col_name] = arr1 * arr2
            correlation_interaction_features.append(col_name)
            
            if ct1 in sig_cell_types and ct2 in sig_cell_types:
                sig_interaction_features.append(col_name)
        except Exception as e:
            print(f"Error creating interaction for {ct1}/{ct2}: {str(e)}")
            continue

    for ct1, ct2 in significant_correlations:
        try:
            col_name = f"{ct1}_{ct2}_logratio"
            arr1 = np.array(freq_df[ct1].values, dtype=float)
            arr2 = np.array(freq_df[ct2].values, dtype=float)
            freq_df[col_name] = np.log(arr1 + 1e-10) - np.log(arr2 + 1e-10)
            correlation_ratio_features.append(col_name)
            
            if ct1 in sig_cell_types and ct2 in sig_cell_types:
                sig_only_ratio_features.append(col_name)
        except Exception as e:
            print(f"Error creating log ratio for {ct1}/{ct2}: {str(e)}")
            continue

    # 4d) Create ratios between ALL cell types (comprehensive set including correlated pairs)
    all_pairs_ratio_features = []
    
    # First include all correlation ratio features
    all_pairs_ratio_features.extend(correlation_ratio_features)
    
    # Then create ratios for any remaining pairs
    for i, ct1 in enumerate(all_cell_types):
        for ct2 in all_cell_types[i+1:]:  # avoid duplicates and self-pairs
            # Check if this pair already has a ratio from correlations
            skip = False
            for existing_ratio in correlation_ratio_features:
                parts = existing_ratio.split('_logratio')[0].split('_')
                if len(parts) == 2 and ((parts[0] == ct1 and parts[1] == ct2) or 
                                        (parts[0] == ct2 and parts[1] == ct1)):
                    skip = True
                    break
            
            # If already in correlation_ratio_features, skip creating a new one
            if skip:
                continue
                
            try:
                # Create a new ratio with unique naming
                col_name = f"{ct1}_{ct2}_allpair_logratio"
                arr1 = np.array(freq_df[ct1].values, dtype=float)
                arr2 = np.array(freq_df[ct2].values, dtype=float)
                freq_df[col_name] = np.log(arr1 + 1e-10) - np.log(arr2 + 1e-10)
                all_pairs_ratio_features.append(col_name)
            except Exception as e:
                print(f"Error creating all-pairs ratio for {ct1}/{ct2}: {str(e)}")
                continue
    
    print(f"Created {len(all_pairs_ratio_features)} ratio features for ALL cell type pairs (including correlation pairs)")
    
    # 4e) Create interaction terms between ALL cell types (comprehensive set including correlated pairs)
    all_pairs_interaction_features = []
    
    # First include all correlation interaction features
    all_pairs_interaction_features.extend(correlation_interaction_features)
    
    # Then create interactions for any remaining pairs
    for i, ct1 in enumerate(all_cell_types):
        for ct2 in all_cell_types[i+1:]:  # avoid duplicates and self-pairs
            # Check if this pair already has an interaction from correlations
            skip = False
            for existing_interaction in correlation_interaction_features:
                parts = existing_interaction.split('_interaction')[0].split('_')
                if len(parts) == 2 and ((parts[0] == ct1 and parts[1] == ct2) or 
                                        (parts[0] == ct2 and parts[1] == ct1)):
                    skip = True
                    break
            
            # If already in correlation_interaction_features, skip creating a new one
            if skip:
                continue
                
            try:
                # Create a new interaction with unique naming
                col_name = f"{ct1}_{ct2}_allpair_interaction"
                arr1 = np.array(freq_df[ct1].values, dtype=float)
                arr2 = np.array(freq_df[ct2].values, dtype=float)
                freq_df[col_name] = arr1 * arr2
                all_pairs_interaction_features.append(col_name)
            except Exception as e:
                print(f"Error creating all-pairs interaction for {ct1}/{ct2}: {str(e)}")
                continue
    
    print(f"Created {len(all_pairs_interaction_features)} interaction features for ALL cell type pairs (including correlation pairs)")

    # 5) Create task-specific derived features
    task_interaction_features = []
    task_ratio_features = []
    task_corr_log_cell_types = []

    if task_specific_correlations:
        # Extract cell types involved in task-specific correlations
        task_corr_cell_types = set()
        for ct1, ct2 in task_specific_correlations:
            task_corr_cell_types.add(ct1)
            task_corr_cell_types.add(ct2)
            
        # Get log-transformed features for these cell types
        task_corr_log_cell_types = [f"{ct}_log" for ct in task_corr_cell_types if f"{ct}_log" in all_log_cell_types]
        
        # Create task-specific interaction and ratio features
        for ct1, ct2 in task_specific_correlations:
            try:
                # Task interaction
                col_name = f"{ct1}_{ct2}_task_interaction"
                arr1 = np.array(freq_df[ct1].values, dtype=float)
                arr2 = np.array(freq_df[ct2].values, dtype=float)
                freq_df[col_name] = arr1 * arr2
                task_interaction_features.append(col_name)
                
                # Task log ratio
                col_name = f"{ct1}_{ct2}_task_logratio"
                freq_df[col_name] = np.log(arr1 + 1e-10) - np.log(arr2 + 1e-10)
                task_ratio_features.append(col_name)
            except Exception as e:
                print(f"Error creating task feature for {ct1}/{ct2}: {str(e)}")
                continue

    # 6) Build feature_sets dict with improved naming and organization
    feature_sets = {
        # Base cell type features
        'all_celltypes': all_log_cell_types,
        'significant_celltypes': sig_log_cell_types,
        
        # Ratio features with clear definitions
        'correlation_pair_ratios': correlation_ratio_features,  # Ratios from correlation pairs
        'significant_celltype_ratios': sig_only_ratio_features,  # Ratios between significant cells
        'all_celltype_pairs_ratios': all_pairs_ratio_features,  # Ratios between all possible cell pairs
        
        # Interaction features
        'correlation_interactions': correlation_interaction_features,  # Interactions from correlation pairs
        'significant_celltype_interactions': sig_interaction_features,  # Interactions between significant cells
        'all_celltype_pairs_interactions': all_pairs_interaction_features,  # Interactions between all possible cell pairs
        
        # Combined feature sets - significant cells 
        'significant_celltypes_with_interactions': sig_log_cell_types + sig_interaction_features,
        'significant_celltypes_with_ratios': sig_log_cell_types + sig_only_ratio_features,
        'significant_celltypes_with_correlation_interactions': sig_log_cell_types + correlation_interaction_features,
        'significant_celltypes_with_correlation_ratios': sig_log_cell_types + correlation_ratio_features,
        
        # Combined feature sets - all cells
        'all_celltypes_with_correlation_interactions': all_log_cell_types + correlation_interaction_features,
        'all_celltypes_with_all_interactions': all_log_cell_types + all_pairs_interaction_features,
        'all_celltypes_with_correlation_ratios': all_log_cell_types + correlation_ratio_features,
        'all_celltypes_with_all_ratios': all_log_cell_types + all_pairs_ratio_features
    }

    # 7) Add task-specific feature sets if applicable
    if task_name and task_specific_correlations:
        feature_sets.update({
            'task_specific_correlation_interactions': task_interaction_features,
            'task_specific_correlation_ratios': task_ratio_features,
            'celltypes_from_task_specific_correlations': task_corr_log_cell_types,
            'significant_celltypes_with_task_specific_correlation_interactions': sig_log_cell_types + task_interaction_features,
            'significant_celltypes_with_task_specific_correlation_ratios': sig_log_cell_types + task_ratio_features
        })
        print(f"Added task-specific feature sets for '{task_name}'")

    # Print feature set summaries
    print("\nFeature Sets Explanation:")
    for name, features in feature_sets.items():
        print(f"- {name}: {len(features)} features")

    if filter_lists and sig_cell_types:
        print(f"\nSignificant cell types after filtering ({len(sig_cell_types)}):")
        for ct in sorted(sig_cell_types):
            print(f"  - {ct}")

    return freq_df, feature_sets

def find_common_features(cytof_freq, scrna_freq, feature_set):
    """
    Identify features from feature_set that can be created in both datasets.
    
    Args:
        cytof_freq: DataFrame with CyTOF frequency data and derived features
        scrna_freq: DataFrame with scRNA-seq frequency data and derived features
        feature_set: List of feature names to check
        
    Returns:
        List of common features present in both datasets
    """
    print(f"Verifying common features from set with {len(feature_set)} features...")
    
    # Since both datasets have already been filtered to common cell types,
    # we just need to check which features from the feature_set exist in both datasets
    common_features = [f for f in feature_set if f in cytof_freq.columns and f in scrna_freq.columns]
    
    # Categorize features by type for reporting
    direct_common = [f for f in common_features if '_log' not in f and '_interaction' not in f and '_logratio' not in f]
    log_common = [f for f in common_features if '_log' in f]
    interaction_common = [f for f in common_features if '_interaction' in f]
    ratio_common = [f for f in common_features if '_logratio' in f]
    
    print(f"Common features breakdown:")
    print(f"  Base features: {len(direct_common)}")
    print(f"  Log-transformed: {len(log_common)}")
    print(f"  Interactions: {len(interaction_common)}")
    print(f"  Log ratios: {len(ratio_common)}")
    print(f"Total common features: {len(common_features)}")
    
    return common_features

def prepare_scrna_validation_data(scrna_freq, cytof_freq, feature_sets):
    """
    Extract and prepare scRNA-seq data for validation,
    ensuring only features common to both datasets are used.
    
    Args:
        scrna_freq: DataFrame with mapped scRNA-seq frequency data
        cytof_freq: DataFrame with CyTOF frequency data
        feature_sets: Dictionary of feature sets from create_feature_sets
        
    Returns:
        Dictionary of common feature sets between CyTOF and scRNA-seq
    """
    print("\nPreparing scRNA-seq data for validation...")
    
    # Create missing derived features in scRNA-seq that exist in CyTOF
    # This is a fix to ensure both datasets have the same derived features
    print("\nCreating derived features (ratios, interactions) in scRNA-seq to match CyTOF...")
    derived_features = [col for col in cytof_freq.columns if '_logratio' in col or '_interaction' in col]
    created_count = 0
    
    for feature in derived_features:
        try:
            if feature in scrna_freq.columns:
                continue  # Skip if already exists
                
            if '_logratio' in feature:
                # Extract the two cell types from the feature name
                parts = feature.split('_logratio')[0].split('_')
                
                # Try to identify the two cell types used in the ratio
                for i in range(1, len(parts)):
                    ct1 = '_'.join(parts[:i])
                    ct2 = '_'.join(parts[i:])
                    
                    # Check if both cell types exist in scRNA-seq data
                    if ct1 in scrna_freq.columns and ct2 in scrna_freq.columns:
                        # Create the log ratio feature
                        arr1 = np.array(scrna_freq[ct1].values, dtype=float)
                        arr2 = np.array(scrna_freq[ct2].values, dtype=float)
                        scrna_freq[feature] = np.log(arr1 + 1e-10) - np.log(arr2 + 1e-10)
                        created_count += 1
                        break
                        
            elif '_interaction' in feature:
                # Extract the two cell types from the feature name
                parts = feature.split('_interaction')[0].split('_')
                
                # Try to identify the two cell types used in the interaction
                for i in range(1, len(parts)):
                    ct1 = '_'.join(parts[:i])
                    ct2 = '_'.join(parts[i:])
                    
                    # Check if both cell types exist in scRNA-seq data
                    if ct1 in scrna_freq.columns and ct2 in scrna_freq.columns:
                        # Create the interaction feature
                        arr1 = np.array(scrna_freq[ct1].values, dtype=float)
                        arr2 = np.array(scrna_freq[ct2].values, dtype=float)
                        scrna_freq[feature] = arr1 * arr2
                        created_count += 1
                        break
                        
        except Exception as e:
            print(f"Error creating derived feature {feature} in scRNA-seq data: {str(e)}")
            continue
            
    print(f"Created {created_count}/{len(derived_features)} derived features in scRNA-seq dataset")
    
    # Identify features available in both datasets
    common_feature_sets = {}
    for fs_name, fs_features in feature_sets.items():
        # Find features that exist in both datasets
        common_features = find_common_features(cytof_freq, scrna_freq, fs_features)
        
        if common_features:
            print(f"Feature set '{fs_name}': {len(common_features)} common features")
            common_feature_sets[fs_name] = common_features
        else:
            print(f"Feature set '{fs_name}': No common features, will be skipped")
    
    return common_feature_sets