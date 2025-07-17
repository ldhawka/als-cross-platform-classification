# ALS Cross-Platform Classification

This repository contains the machine learning pipeline for classifying ALS samples using Random Forest with immune cell type features. Models are trained on CyTOF data and validated on scRNA-seq data, demonstrating cross-platform generalizability.

## Overview

This implementation focuses on two classification tasks from our manuscript:
- **Healthy vs. ALS**: Distinguishing healthy controls from ALS patients  
- **Rapid vs. Non-Rapid**: Classifying ALS progression rates

The approach trains Random Forest models on CyTOF immune profiling data and validates them on an independent scRNA-seq dataset, demonstrating that immune coordination patterns learned from mass cytometry can generalize to single-cell RNA sequencing data.

![Overview of ALS Cross-Platform Classification Pipeline](images/overview_figure.png)


## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Data Requirements

### Required Input Files
To run the classification pipeline, you need the following data files:

1. **CyTOF data**: AnnData (h5ad) format with immune cell type annotations
2. **scRNA-seq data**: AnnData (h5ad) format with immune cell type annotations  
3. **Correlation data**: The provided `correlation_results.csv` file (included in this repository)

### Data Download

The datasets used in this study are available on Zenodo:

**[Download Dataset from Zenodo](https://zenodo.org/record/[PLACEHOLDER-DOI])**

The dataset includes:
- **CyTOF Data** (`als_cytof_data.h5ad`): 2.2 MB - CyTOF immune profiling data from ALS patients and healthy controls
- **scRNA-seq Data** (`Itou2024_scrna_data.h5ad`): 773.60 MB - scRNA-seq validation cohort from independent ALS study

> **Note**: The scRNA-seq data is originally from [Itou et al. 2024](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244263), processed and formatted for cross-platform validation.

### Data Setup Instructions

1. **Download the data files** from Zenodo using the links above
2. **Place them in your preferred location** (e.g., project root or data/ folder)
3. **Update the file paths** in `main.py` (lines 153-154):
   ```python
   # IMPORTANT: Users need to modify these paths to point to their actual data files
   cytof_path = os.path.join(base_dir, "als_cytof_data.h5ad")
   scrna_path = os.path.join(base_dir, "Itou2024_scrna_data.h5ad")
   ```

### Data Format Requirements
Both CyTOF and scRNA-seq datasets must include:
- **`cell_type` column**: Immune cell type annotations using the cell type names compatible with our mapping system
- **`Classifier` column**: Disease state labels (`Healthy`, `Rapid`, `Non-Rapid`)
- **Sample identifier**: For CyTOF data, a `Sample` column; for scRNA-seq data, a `Sample_ID` column (used for frequency calculation)

### Expected Data Structure
```
your_data.h5ad
├── .obs (cell metadata)
│   ├── cell_type          # Cell type annotations
│   ├── Classifier         # Disease state (Healthy/Rapid/Non-Rapid)
│   └── Sample / Sample_ID # Sample identifier for frequency calculation
└── .X (gene expression matrix)
```

### Correlation File
The `correlation_results.csv` file (provided in this repository) contains pre-calculated Spearman correlations between immune cell type frequencies across different progression groups. This file is essential for generating task-specific interaction features that capture coordinated immune dysfunction patterns described in the manuscript. You do not need to modify this file.

## Feature Engineering

The pipeline implements a comprehensive feature engineering approach with **21 different feature sets** that systematically explore various combinations of:

- **Cell type frequencies**: Log-transformed frequencies of immune cell populations
- **Correlation-based interactions**: Interaction terms between cell types with significant correlations  
- **Ratio features**: Log ratios between cell type frequencies
- **Task-specific features**: Features derived from correlations specific to each classification task

For complete details on all 21 feature sets, see `feature_sets_documentation.csv` which provides:
- Feature set names and descriptions
- Calculation methods 
- Component breakdowns
- Examples of feature naming conventions

This systematic approach allows the models to learn from individual cell frequencies, coordinated immune patterns, and task-specific immune signatures.

## Methodology

The classification pipeline implements the approach described in our manuscript:

1. **Cell Type Frequency Calculation**: Extract immune cell type frequencies from both CyTOF and scRNA-seq data
2. **Cross-Platform Cell Type Mapping**: Map cell types between datasets to ensure compatibility  
3. **Feature Engineering**: 
   - Log-transform frequencies for improved distribution normality
   - Generate correlation-based interaction terms using pre-calculated cell-cell correlations
   - Create cell type frequency ratios
4. **Model Training**: Train Random Forest models on CyTOF data using balanced bootstrap sampling
5. **Cross-Platform Validation**: Validate models on independent scRNA-seq data
6. **Feature Importance Analysis**: Extract model-based feature importance (mean decrease in impurity).

## Results

Results are saved to a timestamped directory in the `results/` folder, including:
- **Performance metrics**: Balanced accuracy, F1 score, AUC (CSV format)
- **Feature importance**: Model-based importance scores for all features
- **Predictions**: Model predictions and probabilities on validation data


## Manuscript Alignment

This implementation directly corresponds to the methodology described in our manuscript:
- Uses **Random Forest** as the primary classifier
- Implements **model-based feature importance** (mean decrease in impurity) 
- Focuses on **two main classification tasks**: Healthy vs ALS and Rapid vs Non-Rapid
- Employs **correlation-based interaction features** to capture immune coordination patterns
- Validates models through **cross-platform validation** (CyTOF → scRNA-seq)

## Citation

If you use this code or data, please cite:

**Manuscript:**
```
[Placeholder - Add manuscript citation when published]
```

**Data:**
```
[Placeholder - Add Zenodo data citation]
Dataset: https://zenodo.org/record/[PLACEHOLDER-DOI]
```

**Original scRNA-seq Data:**
```
Itou, T., Fujita, K., Okuzono, Y., Warude, D., Miyakawa, S., Mihara, Y., Matsui, N., Morino, H., Kikukawa, Y., & Izumi, Y. (2024). Th17 and effector CD8 T cells relate to disease progression in amyotrophic lateral sclerosis: A case control study. Journal of Neuroinflammation, 21(1), 331. https://doi.org/10.1186/s12974-024-03327-w Available at: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244263
```