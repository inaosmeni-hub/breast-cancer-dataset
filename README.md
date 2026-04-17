# Breast Cancer Diagnosis — Support Vector Machine (SVM)

**Notebook:** `svm_classification.ipynb`  
**Dataset:** `data.csv` — 569 samples, 30 features, binary target (Malignant / Benign)

## What the notebook covers

- Data loading and preprocessing
- SVM with three kernels: Linear, RBF, Polynomial
- Confusion matrices and classification reports
- ROC curves and AUC comparison
- Hyperparameter tuning with GridSearchCV (C and gamma)
- 5-fold cross-validation
- Feature importance via Linear SVM coefficients
- Decision boundary visualisation using PCA 2D projection
- Summary comparison

## Model results

| Model | Accuracy | AUC |
|---|---|---|
| SVM — linear | 0.9649 | 0.9914 |
| SVM — rbf | 0.9737 | 0.9947 |
| SVM — poly | 0.8860 | 0.9967 |
| SVM — rbf (tuned) | 0.9737 | 0.9947 |

## Observations

1. SVM requires feature scaling before training. Because SVM optimises a margin based on distances between points, features with larger numeric ranges dominate the distance calculations and distort the decision boundary. StandardScaler ensures all 30 features contribute equally.

2. The RBF kernel outperforms both Linear and Polynomial kernels in accuracy and AUC. The boundary between malignant and benign tumours is not perfectly linear in 30-dimensional space, and RBF captures this by mapping the data into a higher-dimensional space where a linear separator exists.

3. The Polynomial kernel achieves the lowest accuracy despite a very high AUC. This means the model ranks predictions well but its probability scores are poorly calibrated, making threshold-based decisions unreliable.

4. GridSearchCV identifies C=1 and gamma='scale' as the optimal hyperparameters — the scikit-learn defaults. This confirms that kernel choice matters more than parameter tuning on this dataset.

5. False negatives are the most critical errors in cancer diagnosis. A missed malignancy delays treatment, making recall on the malignant class the most important metric when evaluating this model.

## Requirements

```
pandas, numpy, matplotlib, scikit-learn
```
