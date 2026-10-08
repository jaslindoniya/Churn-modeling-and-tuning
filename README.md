# Churn-modeling-and-tuning
This project is about predicting customer churn using machine learning.

### What I did in this task
- Created a clean dataset for churn prediction
- Built a preprocessing pipeline (StandardScaler for numbers, OneHotEncoder for categories) to avoid data leakage
- Trained 4 different models:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. LightGBM
- Used GridSearchCV with Stratified K-Fold (3 folds) for hyperparameter tuning
- The tuning was based on F1-Score because churn data is imbalanced

### How I evaluated the models
I checked the models using these metrics:
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC-AUC Curve plot

### Results
| Model | F1-Score | ROC-AUC |
|-------|----------|---------|
| RandomForest | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 |
| LightGBM | 1.00 | 1.00 |
| LogisticRegression | 0.868 | 0.94 |

All tree based models performed very well. The champion model is XGBoost / RandomForest.

### Files in this repo
- `Churn_modeling_and_tuning.ipynb` - Full code with training and evaluation
- `XGBoost_champion_model.pkl` - Saved best model for future use

### Tools Used
Python, Pandas, Scikit-Learn, XGBoost, LightGBM, Matplotlib
