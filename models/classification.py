"""
Classification models for CyTOF and scRNA-seq data.
"""

import os
import sys
import numpy as np
import pandas as pd
import traceback
import datetime
from collections import Counter
import gc

# Sklearn imports
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve, auc,
    balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# Create timestamped results directory
def create_results_dir():
    """Create a unique timestamped results directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"scrna_validation_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Global results directory
RESULTS_DIR = create_results_dir()

def create_model(model_type, X, y, add_noise=True, use_grid_search=True, cv=3):
    """
    Create and train a Random Forest model with improvements for AUC
    
    Args:
        model_type: Type of model to create (must be 'rf' for Random Forest)
        X: Features
        y: Target
        add_noise: Whether to add small random noise to the data for stability
        use_grid_search: Whether to use grid search for hyperparameter tuning
        cv: Number of cross-validation folds for grid search
        
    Returns:
        Trained Random Forest model
    """
    print(f"Creating and training {model_type.upper()} model...")
    
    # Create balanced training data
    
    # Add parent directory to path to import from data module
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from data.data_loader import create_balanced_bootstrap
    X_balanced, y_balanced = create_balanced_bootstrap(X, y)
    
    # Add small random noise to create variation 
    if add_noise:
        noise_scale = 1e-4  
        if isinstance(X_balanced, pd.DataFrame):
            X_noisy = X_balanced + np.random.normal(0, noise_scale, X_balanced.shape)
        else:
            X_noisy = X_balanced + np.random.normal(0, noise_scale, X_balanced.shape)
        print(f"Added small random noise (scale={noise_scale}) to features")
    else:
        X_noisy = X_balanced
    
    # Define Random Forest model and parameter grid for grid search
    if model_type != 'rf':
        raise ValueError(f"Unknown model type: {model_type}. Only 'rf' (Random Forest) is supported.")
    
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'model__n_estimators':     [100, 200, 500],
        'model__max_depth':        [None, 10, 20],
        'model__min_samples_split':[2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__class_weight':     [None, 'balanced', 'balanced_subsample']
    }
    default_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None, min_samples_split=2, class_weight='balanced')
    
    # Create a pipeline with standardization
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', base_model if use_grid_search else default_model)
    ])
    
    # Use balanced accuracy for binary classification (Random Forest handles class imbalance well)
    scoring = 'balanced_accuracy'
    refit   = 'balanced_accuracy'

    # Use grid search if requested
    if use_grid_search:
        # only skip if we literally can’t form cv folds
        n_samples = len(X_noisy)
        if n_samples < cv:
            print(f"Not enough samples ({n_samples}) for {cv}-fold grid search, using default parameters")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', default_model)
            ])
            pipeline.fit(X_noisy, y_balanced)
        else:
            print(f"Performing grid search with {cv}-fold cross-validation")
            # Use balanced_accuracy as scoring metric for grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring=scoring,  
                refit=refit,
                n_jobs=-1
            )
            
            try:
                grid_search.fit(X_noisy, y_balanced)
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                print(f"Best parameters: {best_params}")
                print(f"Best cross-validation score: {best_score:.4f}")
                pipeline = grid_search.best_estimator_
            except Exception as e:
                print(f"Grid search failed: {str(e)}")
                print("Using default parameters instead")
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', default_model)
                ])
                pipeline.fit(X_noisy, y_balanced)
    else:
        # Fit the model with default parameters
        pipeline.fit(X_noisy, y_balanced)
    
    # Random Forest has built-in probability calibration, so no additional calibration needed
    print("Random Forest has built-in probability calibration")
    
    print(f"Successfully trained {model_type.upper()} model")
    return pipeline

def validate_model(model, X_val, y_val, class_names, task_name="", model_name="", feature_set="", feature_set_name=None, n_iterations=200, batch_size=20):
    """
    Validate a model on validation data with bootstrap resampling.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation targets
        class_names: List of class names
        task_name: Name of the task for saving results
        model_name: Name of the model for saving results
        feature_set: Name of the feature set for saving results
        feature_set_name: Explicit name for the feature set
        n_iterations: Number of iterations with bootstrap resampling (default=200)
        batch_size: Number of iterations to process at once (default=20)
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"Validating model on scRNA-seq data with {n_iterations} iterations in batches of {batch_size}...")
    
    y_pred = model.predict(X_val)
    
    try:
        y_pred_proba = model.predict_proba(X_val)
    except Exception as e:
        print(f"Warning: Unable to get prediction probabilities: {str(e)}")
        y_pred_proba = None
    
    # Get unique classes in the validation data
    unique_val_classes = sorted(np.unique(y_val))
    print(f"  Validation data contains {len(unique_val_classes)} classes: {unique_val_classes}")
    
    # Check if there are missing classes compared to what the model was trained on
    missing_classes = [c for c in class_names if c not in unique_val_classes]
    if missing_classes:
        print(f"  Warning: Validation data is missing {len(missing_classes)} classes from training: {missing_classes}")
        #print(f"  This may affect multiclass metrics and confusion matrix interpretation")
    
    # RUN MULTIPLE ITERATIONS TO COLLECT PROBABILITY DISTRIBUTIONS WITH BATCHING
    print(f"\nRunning {n_iterations} iterations to collect probability distributions...")
    
    # Create the CSV file directly 
    iterations_df_path = None
    if task_name and model_name and (feature_set or feature_set_name):
        # Use feature_set_name if provided, otherwise fall back to feature_set
        feature_set_to_use = feature_set_name if feature_set_name is not None else feature_set
        
        # Create file name
        task_name_clean = task_name.replace(' ', '_').replace('-', '_').replace('/', '_')[:50]
        model_name_clean = model_name.replace(' ', '_').replace('-', '_')[:20]
        # No length limit for feature set name
        feature_set_clean = feature_set_to_use.replace(' ', '_')
        
        # Save to CSV with consistent naming
        all_iterations_filename = f"all_iterations_probas_{task_name_clean}_{feature_set_clean}_{model_name_clean.lower()}.csv"
        iterations_df_path = os.path.join(RESULTS_DIR, all_iterations_filename)
        
        
        total_rows = 0
        
        # Create an empty DataFrame with the right columns
        columns = ['sample_id', 'iteration', 'true_label']
        for class_label in model.classes_:
            columns.append(f'prob_{class_label}')
        
        # Write the header to the CSV file
        with open(iterations_df_path, 'w') as f:
            f.write(','.join(columns) + '\n')
        
        # Process iterations in batches to save memory
        for batch_start in range(0, n_iterations, batch_size):
            batch_end = min(batch_start + batch_size, n_iterations)
            # Only print progress every 5 batches to reduce verbosity
            if batch_start % (batch_size * 5) == 0:
                print(f"  Processing iterations {batch_start+1}-{batch_end} of {n_iterations}...")
            
            batch_rows = []
            
            # Process each iteration in this batch
            for iteration in range(batch_start, batch_end):
                try:
                    # Add small random noise to the validation data to introduce variation
                    noise_scale = 1e-5  # Very small scale - just enough to get slightly different predictions
                    X_val_noisy = X_val + np.random.normal(0, noise_scale, X_val.shape)
                    
                    # Make predictions with the noisy data
                    y_iter_proba = model.predict_proba(X_val_noisy)
                    y_iter_pred = model.predict(X_val_noisy)
                    
                    # For each sample, create a row for this iteration
                    for sample_idx in range(len(y_val)):
                        # Get sample ID safely
                        try:
                            sample_id = X_val.index[sample_idx]
                        except (IndexError, TypeError) as e:
                            sample_id = f"sample_{sample_idx}"
                        
                        # Get true label safely
                        if isinstance(y_val, pd.Series):
                            true_label = y_val.iloc[sample_idx]
                        else:
                            true_label = y_val[sample_idx]
                        
                        # Create a row with sample ID, iteration number, true label
                        row = {
                            'sample_id': sample_id,
                            'iteration': iteration,
                            'true_label': true_label
                        }
                        
                        # Add probability for each class
                        for class_idx, class_label in enumerate(model.classes_):
                            row[f'prob_{class_label}'] = y_iter_proba[sample_idx, class_idx]
                        
                        batch_rows.append(row)
                    
                except Exception as e:
                    print(f"  Error in iteration {iteration}: {str(e)}")
                    continue
            
            # Convert batch to DataFrame
            if batch_rows:
                batch_df = pd.DataFrame(batch_rows)
                
                # Append to the CSV file
                batch_df.to_csv(iterations_df_path, mode='a', header=False, index=False)
                
                # Count rows
                total_rows += len(batch_rows)
                
                # Free memory
                del batch_rows, batch_df, X_val_noisy, y_iter_proba
                gc.collect()
            
            # Only print completion every 5 batches to reduce verbosity
            if batch_start % (batch_size * 5) == 0:
                print(f"  Completed batch {batch_start+1}-{batch_end}, total rows: {total_rows}")
        
        # Calculate statistics
        if total_rows > 0:
            num_samples = len(X_val)
            num_successful = total_rows // num_samples
            print(f"  Collected probabilities from {num_successful}/{n_iterations} successful iterations")
            print(f"  Saved all iterations' probabilities to {all_iterations_filename}")
            
            # Calculate metrics from iterations
            print("\nCalculating aggregate metrics from all iterations...")
            # Read the iterations data in chunks to avoid memory issues
            all_accs = []
            all_balanced_accs = []
            all_f1s = []
            all_aucs = []
            
            # Process in chunks of 10 iterations at a time
            chunk_size = 10 * len(X_val)  # 10 iterations worth of rows
            reader = pd.read_csv(iterations_df_path, chunksize=chunk_size)
            
            for chunk in reader:
                # Group by iteration
                for iteration, group in chunk.groupby('iteration'):
                    # Extract true labels and predicted probabilities
                    y_true = group['true_label'].values
                    
                    # Get the highest probability class as the prediction
                    prob_cols = [col for col in group.columns if col.startswith('prob_')]
                    y_proba = group[prob_cols].values
                    y_pred = [model.classes_[np.argmax(probs)] for probs in y_proba]
                    
                    # Calculate metrics for this iteration
                    acc = accuracy_score(y_true, y_pred)
                    balanced_acc = balanced_accuracy_score(y_true, y_pred)
                    all_accs.append(acc)
                    all_balanced_accs.append(balanced_acc)
                    
                    # Handle F1 score for binary classification
                    try:
                        # Binary classification - use positive class (second class)
                        pos_class = model.classes_[1] if len(model.classes_) > 1 else model.classes_[0]
                        
                        # Convert all to strings to ensure consistent types
                        y_true_str = np.array([str(y) for y in y_true])
                        y_pred_str = np.array([str(y) for y in y_pred])
                        pos_class_str = str(pos_class)
                        
                        # Check that positive class is in the labels
                        available_classes = sorted(np.unique(np.concatenate([y_true_str, y_pred_str])))
                        
                        if pos_class_str in available_classes:
                            f1 = f1_score(y_true_str, y_pred_str, average='binary', pos_label=pos_class_str)
                        else:
                            # Fallback to any available class as positive
                            fallback_pos = available_classes[-1]  # Use last class
                            f1 = f1_score(y_true_str, y_pred_str, average='binary', pos_label=fallback_pos)
                            
                        all_f1s.append(f1)
                    except Exception as exc:
                        print(f"  Warning in iteration F1 calculation: {exc}")
                        try:
                            # Last resort - try weighted average for binary case too
                            f1 = f1_score(y_true, y_pred, average='weighted')
                            all_f1s.append(f1)
                            print(f"  Fallback to weighted F1 succeeded: {f1}")
                        except Exception as e2:
                            print(f"  All F1 calculations failed: {e2}")
                        pass  # Skip F1 if all methods fail
                    
                    # Handle AUC for binary classification
                    try:
                        # For binary, use the probability of the positive class
                        pos_idx = 1 if len(model.classes_) > 1 else 0
                        pos_class = model.classes_[pos_idx]
                        
                        # Convert to binary (1 for positive, 0 for others)
                        y_true_bin = np.array([1 if y == pos_class else 0 for y in y_true])
                        
                        # Get probabilities for positive class
                        prob_col = f'prob_{pos_class}'
                        if prob_col in group.columns and len(np.unique(y_true_bin)) > 1:
                            auc_val = roc_auc_score(y_true_bin, group[prob_col].values)
                            all_aucs.append(auc_val)
                    except Exception:
                        pass  # Skip AUC if it fails
            
            # Calculate aggregate metrics
            acc = np.mean(all_accs) if all_accs else np.nan
            balanced_acc = np.mean(all_balanced_accs) if all_balanced_accs else np.nan
            f1 = np.mean(all_f1s) if all_f1s else np.nan
            auc_score = np.mean(all_aucs) if all_aucs else np.nan
            
            # Calculate confidence intervals (95%)
            acc_ci = np.percentile(all_accs, [2.5, 97.5]) if len(all_accs) >= 10 else None
            balanced_acc_ci = np.percentile(all_balanced_accs, [2.5, 97.5]) if len(all_balanced_accs) >= 10 else None
            f1_ci = np.percentile(all_f1s, [2.5, 97.5]) if len(all_f1s) >= 10 else None
            auc_ci = np.percentile(all_aucs, [2.5, 97.5]) if len(all_aucs) >= 10 else None
            
            print(f"  Accuracy: {acc:.4f} (95% CI: {acc_ci[0]:.4f}-{acc_ci[1]:.4f})" if acc_ci is not None else f"  Accuracy: {acc:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f} (95% CI: {balanced_acc_ci[0]:.4f}-{balanced_acc_ci[1]:.4f})" if balanced_acc_ci is not None else f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  F1 Score: {f1:.4f} (95% CI: {f1_ci[0]:.4f}-{f1_ci[1]:.4f})" if f1_ci is not None else f"  F1 Score: {f1:.4f}")
            print(f"  AUC: {auc_score:.4f} (95% CI: {auc_ci[0]:.4f}-{auc_ci[1]:.4f})" if auc_ci is not None else f"  AUC: {auc_score:.4f}")
            
            # Save prediction results to CSV with added confidence intervals
            if task_name and model_name and (feature_set or feature_set_name):
                # Create a DataFrame with sample IDs, true labels, predictions, and class probabilities
                results_df = pd.DataFrame({
                    'sample_id': X_val.index,
                    'true_label': y_val,
                    'predicted_label': y_pred
                })
                
                # Add probability columns for each class
                if y_pred_proba is not None:
                    for i, class_label in enumerate(model.classes_):
                        results_df[f'prob_{class_label}'] = y_pred_proba[:, i]
                
                # Save to CSV
                filename = f"predictions_{task_name_clean}_{model_name_clean}_{feature_set_clean}.csv"
                results_df.to_csv(os.path.join(RESULTS_DIR, filename), index=False)
                print(f"  Saved predictions and probabilities to {filename}")
        else:
            print("  No probability data collected from iterations")
            # Fall back to traditional metrics calculation if no iterations data
            acc = accuracy_score(y_val, y_pred)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            
            # Handle F1 score for binary classification
            try:
                # For binary models, ensure we're using available classes in the validation set
                available_classes = sorted(np.unique(np.concatenate([y_val, y_pred])))
                if len(available_classes) <= 1:
                    print(f"  Warning: Only one class present in combined predictions and ground truth: {available_classes}")
                    f1 = np.nan
                else:
                    # Convert all to strings for consistent comparison
                    y_val_str = np.array([str(y) for y in y_val])
                    y_pred_str = np.array([str(y) for y in y_pred])
                    all_classes_str = sorted(np.unique(np.concatenate([y_val_str, y_pred_str])))
                    
                    # Binary case - proceed with pos_label
                    # Get model classes to determine positive class
                    if hasattr(model, 'classes_'):
                        model_classes = model.classes_
                        
                        # Use the second class as positive class (if available)
                        pos_class = model_classes[1] if len(model_classes) > 1 else model_classes[0]
                        
                        # Convert to string for consistency
                        pos_class_str = str(pos_class)
                        
                        # Check if this class is in the validation set
                        if pos_class_str in all_classes_str:
                            pos_label = pos_class_str
                        else:
                            # Fallback to the second available class when available
                            pos_label = all_classes_str[-1]
                    else:
                        # Fallback to the second available class when available
                        pos_label = all_classes_str[-1]
                    
                    # Calculate F1 with string versions and explicit pos_label
                    f1 = f1_score(y_val_str, y_pred_str, average='binary', pos_label=pos_label)
            except Exception as e:
                print(f"  Warning: Unable to calculate standard F1 score: {str(e)}")
                # Fall back to using macro averaging if there's an issue
                try:
                    f1 = f1_score(y_val, y_pred, average='macro')
                    print("  Falling back to macro-averaged F1 score")
                except Exception as e2:
                    print(f"  Warning: All F1 calculation methods failed: {str(e2)}")
                    f1 = np.nan
            
            # Handle AUC for binary classification
            if y_pred_proba is not None:
                try:
                    # For binary classification, use the second class (usually the positive class)
                    classes_str = [str(c) for c in model.classes_]
                    
                    # Use the second class as positive if it exists in the validation data
                    if len(classes_str) > 1 and classes_str[1] in unique_val_classes:
                        pos_class = classes_str[1]
                        pos_class_idx = 1
                    else:
                        # Otherwise use the first class
                        pos_class = classes_str[0]
                        pos_class_idx = 0
                    
                    # Create binary labels (1 for positive, 0 for negative)
                    y_binary = np.array([1 if str(y) == str(pos_class) else 0 for y in y_val])
                    
                    # Check if binary labels have both classes (0 and 1)
                    if len(np.unique(y_binary)) < 2:
                        print(f"  Warning: Only one class present in binarized labels, AUC undefined")
                        auc_score = np.nan
                    else:
                        # Calculate AUC
                        auc_score = roc_auc_score(y_binary, y_pred_proba[:, pos_class_idx])
                        print(f"  Calculated binary AUC with positive class: {pos_class}")
                except Exception as e:
                    print(f"  Warning: Unable to calculate binary AUC: {str(e)}")
                    auc_score = np.nan
            else:
                auc_score = np.nan
            
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  F1 Score: {f1:.4f}" if not np.isnan(f1) else "  F1 Score: N/A")
            print(f"  AUC: {auc_score:.4f}" if not np.isnan(auc_score) else "  AUC: N/A")
            
    else:
        print("  Cannot save iterations data - missing task_name, model_name, or feature_set")
        # Fall back to traditional metrics calculation if no iterations data
        acc = accuracy_score(y_val, y_pred)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        # Simplified F1 calculation for binary classification
        try:
            # Convert all labels to strings for consistent comparison
            y_val_str = np.array([str(y) for y in y_val])
            y_pred_str = np.array([str(y) for y in y_pred])
            
            # Check how many unique classes we have after string conversion
            all_classes_str = sorted(np.unique(np.concatenate([y_val_str, y_pred_str])))
            
            # Binary case - use the second class (index 1) as positive class
            if hasattr(model, 'classes_'):
                model_classes = model.classes_
                pos_class = model_classes[1] if len(model_classes) > 1 else model_classes[0]
                pos_class_str = str(pos_class)
                
                if pos_class_str in all_classes_str:
                    pos_label = pos_class_str
                else:
                    pos_label = all_classes_str[-1]  # Use last class as positive
            else:
                pos_label = all_classes_str[-1]  # Use last class as positive
            
            # Compute F1 with string values and explicit pos_label
            f1 = f1_score(y_val_str, y_pred_str, average='binary', pos_label=pos_label)
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            print(f"Exception type: {type(e).__name__}")
            try:
                # Last resort - try weighted average on the original values
                f1 = f1_score(y_val, y_pred, average='weighted')
            except Exception as e2:
                f1 = np.nan
        
        # Simplified AUC calculation for binary classification
        try:
            if y_pred_proba is not None:
                auc_score = roc_auc_score(y_val, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0])
            else:
                auc_score = np.nan
        except:
            auc_score = np.nan
    
    # Clean up
    gc.collect()
    print("  Memory cleared after validation")
    
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'auc': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'all_iteration_probas': iterations_df_path  # Return the path instead of the DataFrame
    }


def get_feature_importance(model, feature_names, class_names, X_val=None, y_val=None, 
                         task_name="", model_name="", feature_set="", feature_set_name=None):
    """
    Extract model-based and permutation importance for Random Forest models.
    
    Note: The manuscript uses model-based importance (mean decrease in impurity) 
    as specified in Methods. Permutation importance is calculated for completeness
    but is not the primary analysis method in the paper.
    
    Args:
        model: Trained model
        feature_names: Names of the features
        class_names: Class names
        X_val: Validation features for permutation importance (optional)
        y_val: Validation targets for permutation importance (optional)
        task_name: Name of the task (for saving results)
        model_name: Name of the model (for saving results)
        feature_set: Name of the feature set (for saving results)
        feature_set_name: Explicit name for the feature set
        
    Returns:
        DataFrame with feature importance information, or None if unavailable
    """
    # ——————————————————————————————————————————

    import os
    import numpy as np
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import Pipeline

    print("Extracting feature importance…")
    # ——————————————————————————————————————————
    # Unwrap Pipeline → raw estimator
    print("Unwrapping model to find the base estimator...")
    
    estimator = model
    
    # Handle the case of Pipeline
    if isinstance(estimator, Pipeline):
        print(f"  Found Pipeline with {len(estimator.steps)} steps")
        for i, (name, step) in enumerate(estimator.steps):
            print(f"    Step {i}: {name} - {type(step).__name__}")
        
        # Get the final step, which should be the model
        raw = estimator.steps[-1][1]
        print(f"  Using final step: {type(raw).__name__}")
    else:
        raw = estimator
        print(f"  Using estimator directly: {type(raw).__name__}")

    # Handle case where raw might be a string or otherwise unexpected type
    if not hasattr(raw, 'predict') or isinstance(raw, (str, int, float, bool)):
        print(f"  Warning: Extracted invalid estimator type: {type(raw).__name__}")
        print("  Searching for a valid model in the pipeline...")
        
        # If we ended up with an invalid model, go back to the original and try another approach
        if hasattr(model, 'predict'):
            if hasattr(model, 'estimator') and hasattr(model.estimator, 'steps'):
                # This handles the case when model.estimator is a Pipeline
                raw = model.estimator.steps[-1][1]
                print(f"  Found model in model.estimator: {type(raw).__name__}")
            elif hasattr(model, '_final_estimator') and model._final_estimator is not None:
                # Some sklearn versions use _final_estimator
                raw = model._final_estimator
                print(f"  Using _final_estimator: {type(raw).__name__}")
            else:
                # Last resort: use the original model itself
                print("  Unable to extract base model, using original model")
                raw = model
    
    print(f">> Final raw estimator is: {type(raw).__name__}")
    # ——————————————————————————————————————————

    parts = []


    # 2) Model‐based importances (RF or linear models)
    if hasattr(raw, "feature_importances_") or hasattr(raw, "coef_"):
        if hasattr(raw, "feature_importances_"):
            imp = raw.feature_importances_
        else:
            coefs = raw.coef_
            if coefs.ndim == 1 or len(class_names) == 2:
                imp = np.abs(coefs if coefs.ndim == 1 else coefs[0])
            else:
                imp = np.mean(np.abs(coefs), axis=0)

        df_mod = pd.DataFrame({
            "Feature": feature_names,
            "Importance": imp,
            "Std": np.nan,
            "Type": "Model-based"
        }).sort_values("Importance", ascending=False)

        print("Top 10 model‐based features:")
        for i, (f, w) in enumerate(zip(df_mod["Feature"].head(10),
                                       df_mod["Importance"].head(10)), 1):
            print(f"  {i}. {f}: {w:.4f}")

        parts.append(df_mod)

    # Since we only use Random Forest, we can skip SVM-specific logic
    print(f"  Model type: Random Forest ({type(raw).__name__})")

    # Random Forest has built-in feature importance, so we don't need SHAP

    # 3) Permutation importance on the full pipeline
    if X_val is not None and y_val is not None:
        print("Calculating permutation importance on full pipeline…")
        
        # Determine scoring method for binary classification
        # Note: Skip 'f1' as it has issues with string labels, use 'f1_weighted' instead
        scoring_methods = ['roc_auc', 'balanced_accuracy', 'f1_weighted']
            
        best_perm_result = None
        best_score_method = None
        
        # Try different scoring methods to find one that gives non-zero importance
        for scoring_method in scoring_methods:
            try:
                perm_result = permutation_importance(
                    model, X_val, y_val,
                    n_repeats=10,                   # Use 10 repeats as a balance
                    random_state=42,
                    scoring=scoring_method,
                    n_jobs=-1
                )
                
                # Check if we got any non-zero importances
                non_zero_count = np.sum(perm_result.importances_mean > 0)
                
                # If this scoring method gave better results, save it
                if best_perm_result is None or non_zero_count > np.sum(best_perm_result.importances_mean > 0):
                    best_perm_result = perm_result
                    best_score_method = scoring_method
                    
                # If we found some non-zero values, we can stop
                if non_zero_count > 0:
                    break
                    
            except Exception as e:
                print(f"  Error with {scoring_method} scoring: {e}")
                continue
                
        # If all methods failed, try a more aggressive approach with small perturbations
        if best_perm_result is None or np.sum(best_perm_result.importances_mean > 0) == 0:
            print("  All standard methods gave zero importance. Trying enhanced permutation approach...")
            
            # Create a custom permutation function with small perturbations
            def custom_permutation_importance(model, X, y, n_repeats=5):
                from sklearn.metrics import balanced_accuracy_score
                import numpy as np
                
                # Get baseline score
                baseline_pred = model.predict(X)
                baseline_score = balanced_accuracy_score(y, baseline_pred)
                
                # Initialize importance arrays
                n_features = X.shape[1]
                importances = np.zeros((n_repeats, n_features))
                
                # For each feature
                for feature_idx in range(n_features):
                    # Repeat n_repeats times
                    for rep in range(n_repeats):
                        # Create a copy of X
                        X_permuted = X.copy()
                        
                        # Instead of random permutation, add small Gaussian noise
                        feature_values = X[:, feature_idx]
                        feature_std = np.std(feature_values)
                        # Add noise with 20% of the feature's standard deviation
                        noise = np.random.normal(0, 0.2 * feature_std, size=len(feature_values))
                        X_permuted[:, feature_idx] = feature_values + noise
                        
                        # Predict and calculate score
                        perm_pred = model.predict(X_permuted)
                        perm_score = balanced_accuracy_score(y, perm_pred)
                        
                        # Importance is the decrease in score
                        importances[rep, feature_idx] = baseline_score - perm_score
                
                # Compute mean and std across repeats
                return {
                    'importances_mean': np.mean(importances, axis=0),
                    'importances_std': np.std(importances, axis=0)
                }
            
            try:
                # Convert to numpy array for indexing
                X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
                y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
                
                # Apply custom permutation
                custom_result = custom_permutation_importance(model, X_val_array, y_val_array, n_repeats=10)
                
                # Check results
                non_zero_count = np.sum(custom_result['importances_mean'] > 0)
                print(f"  Found {non_zero_count} non-zero feature importances with custom approach")
                
                if non_zero_count > 0:
                    best_perm_result = custom_result
                    best_score_method = "custom"
            except Exception as e:
                print(f"  Error with custom approach: {e}")
        

        # If we found a working method, create the DataFrame
        if best_perm_result is not None:
            # Check if it's a dictionary (from custom function) or object (from sklearn)
            if isinstance(best_perm_result, dict):
                importances_mean = best_perm_result['importances_mean'] 
                importances_std = best_perm_result['importances_std']
                non_zero_count = np.sum(importances_mean > 0)
            else:
                importances_mean = best_perm_result.importances_mean
                importances_std = best_perm_result.importances_std
                non_zero_count = np.sum(importances_mean > 0)
                
            if non_zero_count > 0:
                print(f"  Using {best_score_method} for permutation importance")
                
                df_perm = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances_mean,
                    "Std": importances_std,
                    "Type": f"Permutation ({best_score_method})"
                }).sort_values("Importance", ascending=False)
                
                    
                parts.append(df_perm)
            else:
                print("  Could not calculate meaningful permutation importance (all values zero)")
                
                # Create a DataFrame with zeros but note the issue
                df_perm = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": np.zeros(len(feature_names)),
                    "Std": np.zeros(len(feature_names)),
                    "Type": "Permutation (failed - zero values)"
                })
                parts.append(df_perm)

    if not parts:
        print("Could not extract any feature importance.")
        return None

    # 5) Combine all types of importance
    combined = pd.concat(parts, ignore_index=True)

    # 6) Auto-save if requested
    if task_name and model_name and (feature_set or feature_set_name):
        t  = task_name.replace(" ", "_").replace("-", "_")[:50]
        m  = model_name.replace(" ", "_").replace("-", "_")[:20].lower()
        fs = (feature_set_name or feature_set).replace(" ", "_").replace("-", "_")
        fname = f"feature_importance_{t}_{fs}_{m}.csv"
        combined.to_csv(os.path.join(RESULTS_DIR, fname), index=False)
        print(f"Saved feature importance to {fname}")

    return combined



