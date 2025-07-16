"""
Main module for ALS cross-platform classification.
Trains Random Forest models on CyTOF data and validates on scRNA-seq data
for Healthy vs ALS and Rapid vs Non-Rapid progression classification.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import datetime
import warnings

# Add the current directory to Python path for direct execution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import modules from the package
from data.data_loader import (
    load_data, prepare_and_filter_data, prepare_data_for_task
)
from features.feature_engineering import (
    create_feature_sets, prepare_scrna_validation_data
)
from models.classification import (
    create_model, validate_model, get_feature_importance, RESULTS_DIR
)
from utils.utils_and_plots import (
    save_feature_importances, print_validation_results,
    save_results_to_csv, find_best_models, save_best_models,
    plot_feature_importance, plot_prediction_distributions,
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve
)



def validate_with_scrna(cytof_freq, mapped_scrna_freq, common_feature_sets, tasks, models=None):
    """
    Train models on CyTOF data and validate on mapped scRNA-seq data

    Args:
        cytof_freq: DataFrame with CyTOF frequency and derived features
        mapped_scrna_freq: DataFrame with scRNA-seq frequency and derived features
        common_feature_sets: dict of feature_name -> list of features common to both
        tasks: list of (task_id, task_name, label_map) tuples
        models: list of model keys ['rf'] (Random Forest only)

    Returns:
        summary_results: dict of results per task and feature set
    """
    import gc
    if models is None:
        models = ['rf']

    summary_results = {}

    for task in tasks:
        task_id, task_name, label_map = task
        summary_results[task_id] = {}

        for fs_name, features in common_feature_sets.items():
            if not features:
                continue
            # ensure features exist
            valid_feats = [f for f in features
                           if f in cytof_freq.columns and f in mapped_scrna_freq.columns]
            if not valid_feats:
                continue

            X_train, y_train, X_val, y_val, class_names = prepare_data_for_task(
                cytof_freq, mapped_scrna_freq, valid_feats, task
            )
            # require at least two classes
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            # remove zero-variance
            variances = X_train.var()
            zero_var = variances[variances == 0].index.tolist()
            if zero_var:
                X_train = X_train.drop(columns=zero_var)
                X_val   = X_val.drop(columns=zero_var)

            summary_results[task_id][fs_name] = {}

            for model_key in models:
                # ----- train & validate -----
                model = create_model(
                    model_key, X_train, y_train,
                    use_grid_search=True, cv=3
                )
                perf = validate_model(
                    model, X_val, y_val, class_names,
                    task_name=task_name,
                    model_name=model_key.upper(),
                    feature_set=fs_name,
                    feature_set_name=fs_name
                )

                # record performance metrics
                summary_results[task_id][fs_name][model_key] = {
                    'accuracy': perf['accuracy'],
                    'balanced_accuracy': perf['balanced_accuracy'],
                    'f1': perf['f1'],
                    'auc': perf['auc']
                }

                # Extract feature importance 
                get_feature_importance(
                    model, valid_feats, class_names,
                    X_val=X_val, y_val=y_val,
                    task_name=task_name,
                    model_name=model_key.upper(),
                    feature_set=fs_name,
                    feature_set_name=fs_name
                )

                # clean up
                del model, perf
                gc.collect()

    return summary_results


def main():
    """Main entry point for the classification pipeline"""
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Configure data paths - users should modify these paths to match their data location
    # Default paths assume data files are in the parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # IMPORTANT: Users need to modify these paths to point to their actual data files
    cytof_path = os.path.join(base_dir, "normalized_data_020125_labeled.h5ad")
    scrna_path = os.path.join(base_dir, "scRNAseq_PBMC/Ito2024_normalized_clean_hvg_labeled.h5ad")
    
    # Check if data files exist
    if not os.path.exists(cytof_path):
        print(f"ERROR: CyTOF data file not found at {cytof_path}")
        print("Please modify the cytof_path variable in main.py to point to your CyTOF data file")
        return
    
    if not os.path.exists(scrna_path):
        print(f"ERROR: scRNA-seq data file not found at {scrna_path}")
        print("Please modify the scrna_path variable in main.py to point to your scRNA-seq data file")
        return
    
    import gc

    # Load and filter data
    cytof_data, scrna_data = load_data(cytof_path, scrna_path)
    cytof_freq_filtered, mapped_scrna_freq_filtered = prepare_and_filter_data(cytof_data, scrna_data)
    del cytof_data, scrna_data
    gc.collect()
    print("Cleared original AnnData objects from memory")

    # Define classification tasks - focusing on the two main tasks for the manuscript
    tasks = [
        ('healthy_vs_als',      'Healthy_vs_ALS',
         lambda x: 'Healthy' if x=='Healthy' else ('ALS' if x in ['Rapid','Non-Rapid'] else None)),
        ('rapid_vs_nonrapid',   'Non_Rapid_vs_Rapid',
         lambda x: 'Non-Rapid' if x=='Non-Rapid' else ('Rapid' if x=='Rapid' else None))
    ]

    # Per-task feature generation and validation
    results = {}
    for task_id, task_name, label_map in tasks:
        print(f"\n{'='*40}\nProcessing task: {task_name}\n{'='*40}")
        cf, fs = create_feature_sets(
            cytof_freq_filtered.copy(), filter_lists=True, task_name=task_name
        )
        sf, _ = create_feature_sets(
            mapped_scrna_freq_filtered.copy(), filter_lists=True, task_name=task_name
        )
        common_fs = prepare_scrna_validation_data(sf, cf, fs)
        task_results = validate_with_scrna(
            cf, sf, common_fs,
            [(task_id, task_name, label_map)],
            models=['rf']
        )
        results.update(task_results)

    # Print and save final results
    print_validation_results(results)
    save_results_to_csv(results, RESULTS_DIR, 'scrna_validation_results.csv')
    best_models = find_best_models(results)
    save_best_models(best_models, RESULTS_DIR, 'best_models.txt')
    del results, best_models
    gc.collect()

    # List generated CSV files
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
    print(f"\nGenerated {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        print(f"  - {f}")

    print(f"\nAnalysis complete. All results saved to {RESULTS_DIR}")
    print("Models trained and evaluated using only cell types common to both datasets")
    print("CSV files were generated for predictions, probabilities, and feature importance metrics.")


if __name__ == '__main__':
    main()
