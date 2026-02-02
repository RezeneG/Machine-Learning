#  Wine Quality Classification with MLP & Comparative Analysis

**Student ID:** 2415644  
**Module:** Machine Learning | Mid-Module Assessment  
**Project Focus:** Building an outstanding, reproducible ML pipeline for binary classification.

## Project Overview

This project implements and rigorously evaluates a **Multilayer Perceptron (MLP)** to classify the quality of red wine as "good" or "bad" based on its physicochemical properties. The work exceeds standard requirements by integrating:
- **Hyperparameter tuning** via Grid Search for model optimisation.
- A **comparative analysis** with Logistic Regression and Random Forest models.
- **Deep exploratory data analysis (EDA)** and feature importance interpretation.
- **Comprehensive evaluation** including error analysis and business-impact insights.

##  Dataset

- **Source:** [UCI Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File:** `winequality-red.csv`
- **License:** Open Data Commons Attribution License (effectively CC BY 4.0)
- **Task:** Binary Classification (Quality ≥ 7 is "good", else "bad")
- **Key Characteristics:** 1,599 samples, 11 features, severe class imbalance (13.6% positive).

##  Quick Start: Run the Full Analysis

1.  **Clone/Download** this project.
2.  **Install dependencies** (creates a consistent environment):
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch the Jupyter Notebook**:
    ```bash
    jupyter notebook MMA_MLP_Project.ipynb
    ```
4.  **Execute all cells** (`Kernel` → `Restart & Run All`). This performs the complete workflow from data loading to final model comparison.

##  Project Structure & Outputs

2415644_MMA_Submission/
│
├── MMA_MLP_Project.ipynb #  Main notebook - Full analysis & code
├── requirements.txt # Exact Python package versions
├── README.md # This file
├── 2415644_MMA_Report.pdf # Concise academic report (~600 words)
│
├── data/ # (Not included) Dataset is downloaded on-the-fly
│
└── outputs/ #  Generated results & visualisations
├── learning_curve.png # Fig 1: MLP training history
├── model_comparison.png # Fig 2 & 3: Performance dashboard & confusion matrices
├── feature_importance.png # Fig 4: Random Forest interpretability
├── correlation_matrix.png # Fig 5: Feature relationships
├── target_distribution.png # Fig 6: Class imbalance visualisation
├── final_model_results.csv #  All model metrics for reporting
└── model_metrics.txt #  Key results for easy copy-paste


##  Key Methodological Highlights

- **Reproducibility:** All random seeds are fixed (NumPy, TensorFlow, Scikit-learn).
- **Rigorous Baseline:** Performance is benchmarked against a **Logistic Regression** model.
- **Model Optimisation:** MLP hyperparameters (layers, regularization, learning rate) are tuned via **GridSearchCV**.
- **Interpretability:** A **Random Forest** provides feature importance, linking results to domain knowledge (e.g., `alcohol` as a key quality indicator).
- **Thorough Evaluation:** Reports Accuracy, Precision, Recall, F1, and ROC-AUC with a focus on the minority class.

## Expected Results

Running the notebook will generate:
1.  **EDA Insights:** Visualisation of class imbalance and key feature correlations.
2.  **Model Performance Table:** A clear comparison of all four models (see example below).
3.  **Publication-Ready Figures:** All six figures required for the report appendix.
4.  **Error Analysis:** A qualitative discussion of model weaknesses and business implications.

## Dependencies

All required Python libraries are listed in `requirements.txt`. The main packages are:
- `tensorflow==2.15.0` (for building & training the MLP)
- `scikit-learn==1.3.2` (for data processing, baselines, and tuning)
- `pandas==2.1.4`, `numpy==1.24.3` (for data manipulation)
- `matplotlib==3.8.2`, `seaborn==0.13.0` (for visualisations)

##  For Further Investigation

The notebook is structured to allow easy extension:
- Experiment with different **MLP architectures** in the tuning grid.
- Test other **imbalance-handling techniques** (e.g., SMOTE, class weighting).
- Use **SHAP values** for deeper neural network interpretability.

---
*This project was developed for academic purposes as part of the Machine Learning mid-module assessment.*