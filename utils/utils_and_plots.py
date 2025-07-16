"""
Utility module for CyTOF and scRNA-seq classification results.

This module contains two types of functions:

1. ESSENTIAL UTILITIES (used by main.py):
   - save_feature_importances: Save feature importance results to CSV
   - print_validation_results: Print summary of results  
   - save_results_to_csv: Export results to CSV files
   - find_best_models: Identify best performing models
   - save_best_models: Save best model information

2. OPTIONAL PLOTTING FUNCTIONS (for additional visualization):
   - plot_feature_importance: Generate feature importance plots
   - plot_prediction_distributions: Generate prediction probability plots

The plotting functions are not required for the main analysis pipeline but can be useful
for generating additional visualizations for papers or presentations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, average_precision_score, 
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def save_feature_importances(feature_importance_results, results_dir):
    """
    Save feature importances to CSV files.
    
    Args:
        feature_importance_results: Dictionary of feature importance results
        results_dir: Directory to save results
    """
    print("\nSaving feature importances to CSV files...")
    
    # Create directory for feature importances
    os.makedirs(os.path.join(results_dir, 'feature_importance'), exist_ok=True)
    
    # For each task
    for task_id, task_results in feature_importance_results.items():
        for feature_set, fs_results in task_results.items():
            for model_type, importance_df in fs_results.items():
                # Skip if None
                if importance_df is None:
                    continue
                
                # Create a clean filename
                clean_task = task_id.replace(' ', '_').replace('/', '_')
                clean_fs = feature_set.replace(' ', '_').replace('/', '_')
                filename = f"feature_importance_{clean_task}_{clean_fs}_{model_type}.csv"
                
                # Save to CSV
                try:
                    filepath = os.path.join(results_dir, 'feature_importance', filename)
                    importance_df.to_csv(filepath, index=False)
                    print(f"  Saved feature importance for {task_id}, {feature_set}, {model_type} to {filepath}")
                except Exception as e:
                    print(f"  Error saving feature importance for {task_id}, {feature_set}, {model_type}: {str(e)}")
    
    print("Feature importances saved successfully.")

def print_validation_results(results):
    """
    Print summary of validation results.
    
    Args:
        results: Dictionary of validation results
    """
    print("\nValidation Results Summary:")
    print("=" * 80)
    
    for task_id, task_results in results.items():
        print(f"\nTask: {task_id}")
        print("-" * 40)
        
        for feature_set, fs_results in task_results.items():
            print(f"\n  Feature Set: {feature_set}")
            
            for model_type, model_results in fs_results.items():
                # Check if we're using the new summary results format or old format
                if 'performance' in model_results:
                    # Old format with 'performance' key
                    perf = model_results['performance']
                else:
                    # New format with metrics directly in the model_results dict
                    perf = model_results
                
                print(f"    {model_type.upper()}:")
                print(f"      Accuracy: {perf['accuracy']:.4f}")
                print(f"      Balanced Accuracy: {perf['balanced_accuracy']:.4f}")
                print(f"      F1 Score: {perf['f1']:.4f}")
                print(f"      AUC: {perf['auc']:.4f}" if not np.isnan(perf['auc']) else "      AUC: N/A")

def save_results_to_csv(results, results_dir, filename='scrna_validation_results.csv'):
    """
    Save validation results to CSV file.
    
    Args:
        results: Dictionary of validation results
        results_dir: Directory to save results
        filename: Name of the CSV file to save
    """
    rows = []
    
    for task_id, task_results in results.items():
        for feature_set, fs_results in task_results.items():
            for model_type, model_results_data in fs_results.items():
                # Check if we're using the new summary results format or old format
                if 'performance' in model_results_data:
                    # Old format with 'performance' key
                    perf = model_results_data['performance']
                else:
                    # New format with metrics directly in the model_results dict
                    perf = model_results_data
                
                row = {
                    'Task': task_id,
                    'Feature_Set': feature_set,
                    'Model': model_type,
                    'Accuracy': perf['accuracy'],
                    'balanced_accuracy': perf['balanced_accuracy'],
                    'F1_Score': perf['f1'],
                    'AUC': perf['auc'] if not np.isnan(perf['auc']) else None
                }
                
                rows.append(row)
    
    # Create DataFrame from rows
    results_df = pd.DataFrame(rows)
    
    for task in results_df['Task'].unique():
        for feature_set in results_df[results_df['Task'] == task]['Feature_Set'].unique():
            filtered_df = results_df[(results_df['Task'] == task) & (results_df['Feature_Set'] == feature_set)]
            
            accuracy_std = filtered_df['Accuracy'].std()
            balanced_accuracy_std = filtered_df['balanced_accuracy'].std()
            f1_std = filtered_df['F1_Score'].std()
            auc_std = filtered_df['AUC'].std()
            
            for idx in filtered_df.index:
                results_df.loc[idx, 'Accuracy_std'] = accuracy_std
                results_df.loc[idx, 'balanced_accuracy_std'] = balanced_accuracy_std
                results_df.loc[idx, 'F1_Score_std'] = f1_std
                results_df.loc[idx, 'AUC_std'] = auc_std
    
    # Save to CSV
    results_df.to_csv(os.path.join(results_dir, filename), index=False)
    print(f"\nResults saved to {os.path.join(results_dir, filename)}")

def find_best_models(results):
    """
    Find best models for each task based on AUC.
    
    Args:
        results: Dictionary of validation results
        
    Returns:
        Dictionary of best model configurations for each task
    """
    best_models = {}
    
    for task_id, task_results in results.items():
        best_auc = -1
        best_config = None
        
        for feature_set, fs_results in task_results.items():
            for model_type, model_results in fs_results.items():
                # Check if we're using the new summary results format or old format
                if 'performance' in model_results:
                    # Old format with 'performance' key
                    perf = model_results['performance']
                else:
                    # New format with metrics directly in the model_results dict
                    perf = model_results
                
                auc_score = perf['auc']
                
                if not np.isnan(auc_score) and auc_score > best_auc:
                    best_auc = auc_score
                    best_config = (feature_set, model_type, perf)
        
        if best_config:
            best_models[task_id] = best_config
    
    # Print best models
    print("\nBest Models by Task (based on AUC):")
    print("=" * 80)
    
    for task_id, (feature_set, model_type, perf) in best_models.items():
        print(f"\nTask: {task_id}")
        print(f"  Best configuration: {feature_set}, {model_type.upper()}")
        print(f"  AUC: {perf['auc']:.4f}" if not np.isnan(perf['auc']) else "  AUC: N/A")
        print(f"  Balanced Accuracy: {perf['balanced_accuracy']:.4f}")
        print(f"  Accuracy: {perf['accuracy']:.4f}")
        print(f"  F1 Score: {perf['f1']:.4f}")
    
    return best_models

def save_best_models(best_models, results_dir, filename='best_models.txt'):
    """
    Save best models information to a text file.
    
    Args:
        best_models: Dictionary of best model configurations for each task
        results_dir: Directory to save results
        filename: Name of the text file to save
    """
    with open(os.path.join(results_dir, filename), 'w') as f:
        f.write("Best Models by Task (based on AUC):\n")
        f.write("=" * 80 + "\n\n")
        
        for task_id, (feature_set, model_type, perf) in best_models.items():
            f.write(f"Task: {task_id}\n")
            f.write(f"  Best configuration: {feature_set}, {model_type.upper()}\n")
            
            if not np.isnan(perf['auc']):
                f.write(f"  AUC: {perf['auc']:.4f}\n")
            else:
                f.write("  AUC: N/A\n")
                
            f.write(f"  Balanced Accuracy: {perf['balanced_accuracy']:.4f}\n")
            f.write(f"  Accuracy: {perf['accuracy']:.4f}\n")
            f.write(f"  F1 Score: {perf['f1']:.4f}\n")
            
            f.write("\n")
    
    print(f"Best models information saved to {os.path.join(results_dir, filename)}")

def plot_feature_importance(feature_importance, task_name, model_name="", feature_set="", results_dir=None, n_top=20):
    """
    Plot feature importance from a model.
    
    Args:
        feature_importance: DataFrame with feature importance information
        task_name: Name of the task for the plot title
        model_name: Name of the model for the filename
        feature_set: Name of the feature set for the filename
        results_dir: Directory to save results
        n_top: Number of top features to plot
    """
    if feature_importance is None:
        print("No feature importance data available")
        return
    
    # Process feature types separately
    model_importance = feature_importance[feature_importance['Type'] == 'Model-based']
    perm_importance = feature_importance[feature_importance['Type'] == 'Permutation']
    
    # Create cleaned filenames
    task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20] if model_name else ""
    feature_set_clean = feature_set.replace(' ', '_').replace('-', '_')[:20] if feature_set else ""
    
    filename_base = f"feature_importance_{task_name_clean}"
    if model_name_clean:
        filename_base += f"_{model_name_clean}"
    if feature_set_clean:
        filename_base += f"_{feature_set_clean}"
    
    # Plot model-based importance
    if not model_importance.empty:
        plt.figure(figsize=(12, 10))
        top_features = model_importance.sort_values('Importance', ascending=False).head(n_top)
        
        # Plot horizontal bar chart
        ax = sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {n_top} Features (Model-based) - {task_name}')
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(results_dir, f"{filename_base}_model_based.png"))
            plt.close()
        else:
            plt.show()
    
    # Plot permutation importance if available
    if not perm_importance.empty:
        plt.figure(figsize=(12, 10))
        top_features = perm_importance.sort_values('Importance', ascending=False).head(n_top)
        
        # Plot horizontal bar chart with error bars
        ax = sns.barplot(x='Importance', y='Feature', data=top_features)
        
        # Add error bars if std is available
        if 'Std' in top_features.columns and not top_features['Std'].isna().all():
            x_coords = top_features['Importance'].values
            y_coords = range(len(top_features))
            xerr = top_features['Std'].values
            plt.errorbar(x_coords, y_coords, xerr=xerr, fmt='none', ecolor='black', capsize=5)
        
        plt.title(f'Top {n_top} Features (Permutation) - {task_name}')
        plt.tight_layout()
        
        if results_dir:
            plt.savefig(os.path.join(results_dir, f"{filename_base}_permutation.png"))
            plt.close()
        else:
            plt.show()

def plot_prediction_distributions(y_true, y_pred_proba, classes, task_name, model_name="", feature_set="", results_dir=None):
    """
    Plot prediction probability distributions for each class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: Class names
        task_name: Name of the task for the plot title
        model_name: Name of the model for the filename
        feature_set: Name of the feature set for the filename
        results_dir: Directory to save results
    """
    if y_pred_proba is None:
        print("No prediction probabilities available")
        return
    
    # Create cleaned filenames
    task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20] if model_name else ""
    feature_set_clean = feature_set.replace(' ', '_').replace('-', '_')[:20] if feature_set else ""
    
    filename_base = f"prob_dist_{task_name_clean}"
    if model_name_clean:
        filename_base += f"_{model_name_clean}"
    if feature_set_clean:
        filename_base += f"_{feature_set_clean}"
    
    # Convert classes and labels to strings for consistency
    classes_str = [str(c) for c in classes]
    y_true_str = np.array([str(x) for x in y_true])
    
    # Create a DataFrame with true labels and predicted probabilities
    prob_df = pd.DataFrame({
        'true_label': y_true_str
    })
    
    # Add class probabilities
    for i, class_name in enumerate(classes_str):
        if i < y_pred_proba.shape[1]:
            prob_df[f'prob_{class_name}'] = y_pred_proba[:, i]
    
    # Plot probability distributions for each class
    plt.figure(figsize=(12, 10))
    n_classes = len(classes_str)
    
    # Calculate number of rows and columns for subplot grid
    n_rows = (n_classes + 1) // 2  # Ceiling division
    n_cols = min(2, n_classes)
    
    for i, class_name in enumerate(classes_str):
        if i < y_pred_proba.shape[1]:
            plt.subplot(n_rows, n_cols, i+1)
            
            # Plot probability distribution for true positives and true negatives
            true_pos = prob_df[prob_df['true_label'] == class_name][f'prob_{class_name}']
            true_neg = prob_df[prob_df['true_label'] != class_name][f'prob_{class_name}']
            
            if len(true_pos) > 0:
                sns.histplot(true_pos, color='green', alpha=0.5, label=f'True {class_name}', kde=True)
            if len(true_neg) > 0:
                sns.histplot(true_neg, color='red', alpha=0.5, label=f'Not {class_name}', kde=True)
            
            plt.title(f'Probability Distribution for Class: {class_name}')
            plt.xlabel(f'Predicted Probability of {class_name}')
            plt.ylabel('Count')
            plt.legend()
    
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(os.path.join(results_dir, f"{filename_base}.png"))
        plt.close()
    else:
        plt.show()
    
    # Also save the probabilities as CSV for further analysis
    if results_dir:
        prob_df.to_csv(os.path.join(results_dir, f"{filename_base}.csv"), index=False)
        print(f"Prediction probabilities saved to {os.path.join(results_dir, f'{filename_base}.csv')}")

def plot_confusion_matrix(y_true, y_pred, classes, task_name, model_name="", task_type="binary", normalize=True, results_dir=None):
    """
    Plot confusion matrix for classification results.
    Handles string class labels directly without integer conversion.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        task_name: Name of the task for the plot title
        model_name: Name of the model for the filename
        task_type: Type of task ('binary')
        normalize: Whether to normalize the confusion matrix
        results_dir: Directory to save results (if None, uses current directory)
    """
    # Create a simpler, more reliable filename
    task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20] if model_name else ""
    
    filename_base = f"cm_{task_name_clean}"
    if model_name_clean:
        filename_base += f"_{model_name_clean}"
    
    # Skip if there's no meaningful data to plot
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No data to plot confusion matrix with")
        return
    
    # Keep values as strings to avoid conversion issues
    y_true_arr = np.array([str(x) for x in y_true])
    y_pred_arr = np.array([str(x) for x in y_pred])
    
    # Get unique classes from the data
    unique_classes = sorted(set(np.unique(y_true_arr)) | set(np.unique(y_pred_arr)))
    
    # Convert class list to strings as well for comparison
    classes_str = [str(c) for c in classes]
    
    # Determine which classes to use - prefer the provided classes if they match with the data
    if set(unique_classes).issubset(set(classes_str)):
        display_classes = classes_str
        # Only use classes that appear in the data
        display_classes = [c for c in classes_str if c in unique_classes]
    else:
        print(f"Warning: Class mismatch between data {unique_classes} and provided classes {classes_str}")
        # Use classes from the data if we don't have a good match
        display_classes = unique_classes
    
    # Plot binary confusion matrix
    print(f"Plotting binary confusion matrix with classes: {display_classes}")
    
    try:
        # Create confusion matrix using string labels directly
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=display_classes)
        
        plt.figure(figsize=(10, 8))
        
        if normalize:
            # Check for zero rows to avoid division by zero
            row_sums = cm.sum(axis=1)
            # Replace zeros with ones to avoid division by zero
            row_sums[row_sums == 0] = 1
            
            # Normalize by row (true labels)
            cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
            
            # Use percentage format for normalized matrix
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                        xticklabels=display_classes, yticklabels=display_classes)
            plt.title(f'Confusion Matrix (normalized) - {task_name}')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=display_classes, yticklabels=display_classes)
            plt.title(f'Confusion Matrix - {task_name}')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save with a clean, reliable path
        save_dir = results_dir if results_dir else os.getcwd()
        plt.savefig(os.path.join(save_dir, f"{filename_base}.png"))
        plt.close()  # Close instead of show to avoid blocking
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        # Print debug info
        print(f"y_true unique values: {np.unique(y_true_arr)}")
        print(f"y_pred unique values: {np.unique(y_pred_arr)}")
        print(f"display_classes: {display_classes}")
        
        # Try a simplified version as a last resort
        try:
            print("Attempting simplified confusion matrix...")
            plt.figure(figsize=(8, 6))
            # Just use whatever classes are in the data
            cm = confusion_matrix(y_true_arr, y_pred_arr)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Basic Confusion Matrix - {task_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            save_dir = results_dir if results_dir else os.getcwd()
            plt.savefig(os.path.join(save_dir, f'basic_cm_{task_name_clean}.png'))
            plt.close()
            print("Simplified confusion matrix saved")
        except Exception as e2:
            print(f"Even simplified confusion matrix failed: {e2}")

def plot_roc_curve(y_true, y_pred_proba, classes, task_name, model_name="", feature_set="", feature_set_name=None, model=None, results_dir=None):
    """
    Plot ROC curve for classification results with improved robustness.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: Class names
        task_name: Name of the task for the plot title
        model_name: Name of the model for the filename
        feature_set: Name of the feature set for the filename - kept for backwards compatibility
        feature_set_name: Explicit name for the feature set, preferred over feature_set parameter
        model: Model object to extract class information
        results_dir: Directory to save results (if None, uses current directory)
    """
    # Use feature_set_name if provided, otherwise fall back to feature_set
    feature_set_to_use = feature_set_name if feature_set_name is not None else feature_set
    
    # Create a simpler, more reliable filename
    task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20] if model_name else ""
    # No length limit for feature set name
    feature_set_clean = feature_set_to_use.replace(' ', '_').replace('-', '_') if feature_set_to_use else ""
    
    filename_base = f"roc_{task_name_clean}"
    if model_name_clean:
        filename_base += f"_{model_name_clean}"
    if feature_set_clean:
        filename_base += f"_{feature_set_clean}"

    plt.figure(figsize=(10, 8))
    
    # Convert classes and labels to strings for consistency
    classes_str = [str(c) for c in classes]
    y_true_str = np.array([str(x) for x in y_true])
    
    try:
        if len(classes) == 2:
            # Binary classification case
            pos_class = classes_str[1]  # Fix: Use second class consistently as positive (ALS, Rapid, etc.)
            
            # Find the column index for the positive class
            try:
                # First try using the model directly if available
                if model is not None and hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    model_obj = model.named_steps['model']
                    if hasattr(model_obj, 'classes_'):
                        model_classes = [str(c) for c in model_obj.classes_]
                        pos_index = model_classes.index(pos_class)
                    else:
                        # Fallback to second class
                        pos_index = 1 if len(classes_str) > 1 else 0
                elif model is not None and hasattr(model, 'classes_'):
                    model_classes = [str(c) for c in model.classes_]
                    pos_index = model_classes.index(pos_class)
                else:
                    # Use 1 as default (second column of probabilities)
                    pos_index = 1 if y_pred_proba.shape[1] > 1 else 0
                
                # Create binary labels where 1 = positive class
                y_binary = np.array([1 if y == pos_class else 0 for y in y_true_str])
                
                # Calculate ROC - use first column if only one class
                prob_col = y_pred_proba[:, pos_index] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
                fpr, tpr, _ = roc_curve(y_binary, prob_col, pos_label=1)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                
            except Exception as e:
                print(f"Error in binary ROC curve calculation: {e}")
                # Fallback method 
                try:
                    # If we don't know which class is positive, just use binary indicator
                    y_binary = np.array([1 if y == y_true_str[0] else 0 for y in y_true_str])
                    fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, 0], pos_label=1)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                except Exception as e2:
                    print(f"Fallback ROC curve also failed: {e2}")
                    return
                
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")
        return
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {task_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save figure and close instead of show
    save_dir = results_dir if results_dir else os.getcwd()
    plt.savefig(os.path.join(save_dir, f"{filename_base}.png"))
    plt.close()  # Use close instead of show to avoid blocking

def plot_pr_curve(y_true, y_pred_proba, classes, task_name, model_name="", feature_set="", feature_set_name=None, model=None, results_dir=None):
    """
    Plot precision-recall curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        classes: Class names
        task_name: Name of the task for the plot title
        model_name: Name of the model for the filename
        feature_set: Name of the feature set for the filename - kept for backwards compatibility
        feature_set_name: Explicit name for the feature set, preferred over feature_set parameter
        model: Model object to extract class information
        results_dir: Directory to save results (if None, uses current directory)
    """
    # Use feature_set_name if provided, otherwise fall back to feature_set
    feature_set_to_use = feature_set_name if feature_set_name is not None else feature_set
    
    # Create a simpler, more reliable filename
    task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
    model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20] if model_name else ""
    # No length limit for feature set name
    feature_set_clean = feature_set_to_use.replace(' ', '_').replace('-', '_') if feature_set_to_use else ""
    
    filename_base = f"pr_{task_name_clean}"
    if model_name_clean:
        filename_base += f"_{model_name_clean}"
    if feature_set_clean:
        filename_base += f"_{feature_set_clean}"
    
    plt.figure(figsize=(10, 8))
    
    # Convert classes and labels to strings for consistency
    classes_str = [str(c) for c in classes]
    y_true_str = np.array([str(x) for x in y_true])
    
    try:
        if len(classes_str) == 2:
            # Binary classification case
            pos_class = classes_str[1]  # Fix: Use second class consistently as positive (ALS, Rapid, etc.)
            
            # Create binary labels where 1 = positive class
            y_binary = np.array([1 if y == pos_class else 0 for y in y_true_str])
            
            try:
                # First try using the model directly if available
                if model is not None and hasattr(model, 'named_steps') and 'model' in model.named_steps:
                    model_obj = model.named_steps['model']
                    if hasattr(model_obj, 'classes_'):
                        model_classes = [str(c) for c in model_obj.classes_]
                        pos_index = model_classes.index(pos_class)
                    else:
                        # Fallback to second class
                        pos_index = 1 if len(classes_str) > 1 else 0
                elif model is not None and hasattr(model, 'classes_'):
                    model_classes = [str(c) for c in model.classes_]
                    pos_index = model_classes.index(pos_class)
                else:
                    # Use 1 as default (second column of probabilities)
                    pos_index = 1 if y_pred_proba.shape[1] > 1 else 0
                
                # Calculate PR curve
                prob_col = y_pred_proba[:, pos_index] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
                precision, recall, _ = precision_recall_curve(y_binary, prob_col)
                avg_precision = average_precision_score(y_binary, prob_col)
                baseline = np.sum(y_binary) / len(y_binary)
                
                plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
                plt.axhline(y=baseline, color='r', linestyle='--', label=f'No Skill (AP = {baseline:.2f})')
                
            except Exception as e:
                print(f"Error in binary PR curve calculation: {e}")
                # Fall back to simpler method
                try:
                    # If we don't know which class is positive, just use binary indicator
                    y_binary = np.array([1 if y == y_true_str[0] else 0 for y in y_true_str])
                    precision, recall, _ = precision_recall_curve(y_binary, y_pred_proba[:, 0])
                    avg_precision = average_precision_score(y_binary, y_pred_proba[:, 0])
                    baseline = np.sum(y_binary) / len(y_binary)
                    
                    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
                    plt.axhline(y=baseline, color='r', linestyle='--', label=f'No Skill (AP = {baseline:.2f})')
                except Exception as e2:
                    print(f"Fallback PR curve also failed: {e2}")
                    return
                
    except Exception as e:
        print(f"Error plotting PR curve: {e}")
        return
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {task_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    # Save figure and close instead of show
    save_dir = results_dir if results_dir else os.getcwd()
    plt.savefig(os.path.join(save_dir, f"{filename_base}.png"))
    plt.close()  # Use close instead of show to avoid blocking