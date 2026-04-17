# Breast Cancer Diagnosis — Support Vector Machine (SVM)

## Abstract

This notebook applies Support Vector Machine classification to the Wisconsin Breast Cancer dataset — 569 tumour samples described by 30 numeric features derived from cell nucleus measurements, labelled as Malignant or Benign. Three SVM kernel functions are compared: Linear, RBF, and Polynomial. The best-performing kernel is tuned with GridSearchCV over the C and gamma parameters and evaluated with 5-fold cross-validation. Results are reported through classification reports, confusion matrices, ROC curves, feature importance plots, and a PCA decision boundary visualisation.

---

## Written Observations

**1. Feature Scaling is Not Optional for SVM**
SVM maximises the margin between classes by finding a hyperplane defined by support vectors — the training points closest to the boundary. This optimisation is entirely distance-based, which means features with larger numeric ranges exert disproportionate influence on the margin calculation. In this dataset, features such as `area_mean` range into the hundreds while `fractal_dimension_mean` values are near zero. Without StandardScaler normalisation, the SVM would effectively ignore the small-range features entirely. Scaling to zero mean and unit variance ensures all 30 morphological measurements contribute equally to the decision boundary.

**2. RBF Kernel is Best Suited to This Dataset**
The RBF kernel achieves the highest accuracy (0.9737) among the three kernel configurations. This is consistent with the nature of the data: the boundary between malignant and benign tumours in 30-dimensional feature space is unlikely to be perfectly linear, and the polynomial kernel introduces boundary shapes that may not align with the true geometry of the data. RBF avoids both of these limitations by implicitly mapping the data into an infinite-dimensional feature space where a linear separator exists, making it the most flexible and generally applicable kernel for high-dimensional tabular data.

**3. High AUC Despite Low Accuracy in the Polynomial Kernel Reveals Calibration Problems**
The Polynomial kernel produces the lowest accuracy (0.8860) but the highest AUC (0.9967) among the three kernels. This apparent contradiction arises because accuracy measures how often the model places a prediction on the correct side of the 0.5 threshold, while AUC measures how well the model ranks positives above negatives across all thresholds. A model can rank predictions nearly perfectly while still placing them poorly relative to the 0.5 boundary — indicating poor probability calibration. In a clinical screening context this is a serious problem, because the threshold-based decision (flag as malignant or not) depends on calibrated probabilities.

**4. The Default RBF Configuration is Already Near-Optimal for This Dataset**
GridSearchCV over C ∈ {0.1, 1, 10, 100} and gamma ∈ {scale, auto, 0.001, 0.01, 0.1} selects C=1 and gamma='scale' — exactly the scikit-learn defaults. This result tells us that the primary source of performance variation on this dataset is the kernel choice, not the regularisation or bandwidth parameters. It also demonstrates the value of validation before tuning: running a grid search confirmed that the default hyperparameters are appropriate, which would otherwise be assumed rather than verified.

**5. Minimising False Negatives is the Clinical Priority**
In breast cancer screening, a false negative — a malignant tumour classified as benign — is substantially more harmful than a false positive. A missed malignancy delays diagnosis and treatment, potentially at the cost of patient survival. A false positive leads to additional testing, which causes anxiety but is clinically recoverable. The SVM RBF model achieves high recall on the malignant class, meaning it correctly identifies the large majority of malignant cases. In a real deployment, the classification threshold could be lowered below 0.5 to further increase malignant recall at the cost of additional false positives — a trade-off that clinical teams would calibrate based on the specific screening context.

---

*Keywords: SVM, support vector machine, breast cancer, binary classification, RBF kernel, linear kernel, polynomial kernel, GridSearchCV, cross-validation, feature scaling, StandardScaler, PCA, decision boundary, AUC, false negatives*
