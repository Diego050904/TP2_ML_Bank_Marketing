# TP2 - Supervised Classification: Bank Marketing

72.75 Aprendizaje Automático (Machine Learning) - ITBA 2026 Q2

Prediction of term deposit subscription (`y`) from the Bank Marketing dataset
([Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)).

## Structure
- `data/`: raw dataset and train/test splits
- `notebooks/`: analysis and models, in execution order
- `figures/`: plots for the presentation

## Notebooks
1. `01_preprocessing_and_eda.ipynb`: data cleaning, 80/20 split, EDA and feature decisions
2. `02_classification_cv.ipynb`: Naive Bayes, SVM, KNN and Random Forest evaluated with stratified 5-fold cross-validation (ROC-AUC and F1)
3. `03_hyperparameter_tuning.ipynb`: validation curves for SVM (C), KNN (n_neighbors) and Random Forest (max_depth)
4. `04_final_model.ipynb`: final Random Forest trained on the full train set and evaluated once on the test set

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
