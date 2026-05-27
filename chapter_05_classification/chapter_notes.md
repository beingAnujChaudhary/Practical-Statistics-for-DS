# Chapter 05: Classification

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Classification is a predictive modelling technique used when the target variable belongs to categories or classes. It is a fundamental form of supervised learning where we train a model on labelled data (where the outcome is known) and then apply it to new data where the outcome is unknown.

Unlike regression, which predicts continuous values, classification outputs discrete class labels (e.g., spam/not spam, default/paid off, click/no click). Often, classification models also output a probability score (propensity) that can be thresholded to make decisions, providing flexibility in balancing precision and recall based on business needs.

**Examples of classification problems:**
- Will a customer churn?
- Is an email spam or not spam?
- Will a flight be delayed?
- Is a transaction fraudulent?
- Which product category will a user prefer?

**This chapter introduces:**
- Classification fundamentals
- Naive Bayes
- Discriminant analysis (LDA, QDA)
- Logistic regression
- K-nearest neighbours (KNN)
- Classification performance metrics (confusion matrix, precision, recall, ROC, AUC)
- Strategies for handling imbalanced classes

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand classification problems and distinguish them from regression
- Apply Naive Bayes classification with categorical and numeric predictors
- Understand and implement Linear Discriminant Analysis (LDA)
- Fit, interpret, and evaluate logistic regression models
- Understand odds, log-odds, and probability relationships
- Explain and apply K-nearest neighbours (KNN)
- Evaluate classification models using appropriate metrics
- Construct and interpret confusion matrices, ROC curves, and lift charts
- Recognise and address the rare class problem
- Apply strategies for imbalanced data: undersampling, oversampling, weighting, SMOTE

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Classification | Predict categorical outcomes | `LogisticRegression()`, `GaussianNB()`, etc. |
| Binary Classification | Two classes (0/1, Yes/No) | All sklearn classifiers |
| Multi-class Classification | More than two classes | All sklearn classifiers (multi-class support) |
| Logistic Regression | Probability-based classifier | `LogisticRegression()` |
| Odds | Ratio of success probability to failure probability | Conceptual |
| Log-Odds (Logit) | Logarithmic transformation of odds | Core of logistic regression |
| Naive Bayes | Probability-based classifier with independence assumption | `GaussianNB()`, `MultinomialNB()` |
| Linear Discriminant Analysis (LDA) | Models group separation with normality assumption | `LinearDiscriminantAnalysis()` |
| K-Nearest Neighbours (KNN) | Instance-based classification using nearest examples | `KNeighborsClassifier()` |
| Confusion Matrix | Tabular evaluation of predictions | `confusion_matrix()` |
| Accuracy | Overall correctness (TP + TN) / Total | `accuracy_score()` |
| Precision | Proportion of predicted positives that are correct | `precision_score()` |
| Recall (Sensitivity) | Proportion of actual positives correctly identified | `recall_score()` |
| Specificity | Proportion of actual negatives correctly identified | Derived from confusion matrix |
| F1-Score | Harmonic mean of precision and recall | `f1_score()` |
| ROC Curve | Plot of TPR vs. FPR across thresholds | `roc_curve()` |
| AUC | Area under the ROC curve | `roc_auc_score()` |
| Lift Chart | Improvement over random selection | Custom calculation |
| Class Imbalance | When one class is much rarer than the other | `imblearn` package |

---

## Key Statistical Ideas

### 1. What Is Classification?

Classification predicts **discrete categories** instead of numerical values.

| Problem | Type | Classes |
|---|---|---|
| Spam detection | Binary | Spam / Not Spam |
| Disease diagnosis | Binary | Positive / Negative |
| Flight delay category | Multi-class | On Time / Delayed / Cancelled |
| Product recommendation | Multi-class | Category A / B / C |

**Key Difference from Regression:**

| | Regression | Classification |
|---|---|---|
| **Output** | Continuous values | Discrete categories |
| **Example** | House price = 500,000 | Spam / Not Spam |
| **Evaluation** | RMSE, R² | Accuracy, Precision, Recall, AUC |

---

### 2. Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with a "naive" assumption of conditional independence among predictors.

**Bayes' Theorem:**
$$P(Y=i \mid X_1, \dots, X_p) = \frac{P(Y=i) \prod_{j=1}^{p} P(X_j \mid Y=i)}{\sum_{k} P(Y=k) \prod_{j=1}^{p} P(X_j \mid Y=k)}$$

**Algorithm:**
1. For each class $i$, estimate individual conditional probabilities $P(X_j \mid Y=i)$ from training data.
2. Multiply these probabilities together, along with the prior $P(Y=i)$.
3. Normalise across all classes to get posterior probabilities.
4. Assign the record to the class with the highest posterior probability.

**Key Properties:**
- Works naturally with categorical predictors; numeric predictors must be binned or modelled with parametric distributions (e.g., Gaussian).
- Fast to train and score; scales well to large datasets.
- Often produces biased probability estimates but good ranking for classification.
- Despite the unrealistic independence assumption, it performs surprisingly well in practice, especially for text classification and spam detection.

**Laplace Smoothing:** Add a small constant ($\alpha$) to counts to avoid zero probabilities when a predictor-category combination is absent in training data.

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# For numeric features (Gaussian assumption)
model = GaussianNB()
model.fit(X_train, y_train)

# For categorical/text features (multinomial)
model = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing
model.fit(X_train, y_train)

```

---

### 3. Discriminant Analysis

**Linear Discriminant Analysis (LDA):**

* Assumes predictors are normally distributed within each class with **equal covariance matrices**.
* Finds the linear combination of predictors that maximises separation between class means relative to within-class variance.
* The decision boundary is linear in predictor space.
* Fisher's Linear Discriminant maximises: $\frac{SS_{\text{between}}}{SS_{\text{within}}}$.

**Quadratic Discriminant Analysis (QDA):**

* Relaxes the equal covariance assumption; allows class-specific covariance matrices.
* The decision boundary is quadratic; more flexible but requires more data.

**Key Terms:**

* **Covariance matrix:** Measures how predictors vary together; central to LDA/QDA.
* **Discriminant function:** Linear combination of predictors used for classification.
* **Mahalanobis distance:** Distance metric accounting for covariance; used in LDA.

**Discriminant Function:**


$$\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log(\pi_k)$$


Assign to class with largest $\delta_k(\mathbf{x})$.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

```

---

### 4. Logistic Regression

Despite its name, **logistic regression is a classification algorithm**. It predicts the **probability of class membership**.

**Core Transformation (Logit):** Map probability $p \in [0, 1]$ to log-odds $\in (-\infty, +\infty)$:

$$\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Inverse (Logistic/Sigmoid Function):**

$$P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}}$$

**Why Log-Odds?** Probabilities are bounded $[0, 1]$, while log-odds are unbounded $(-\infty, +\infty)$, making them suitable for linear modelling.

**Fitting:** Maximum Likelihood Estimation (MLE), not least squares. Uses iterative optimisation (e.g., Newton-Raphson).

**Coefficient Interpretation:**

* $\beta_j$ = change in log-odds per unit change in $X_j$, holding other predictors constant.
* **Odds Ratio** = $e^{\beta_j}$ = multiplicative change in odds per unit change in $X_j$.
* *Example:* $\beta = 0.693$ → odds ratio = 2.0 → doubling of odds.

**Generalised Linear Model (GLM):** Logistic regression is a GLM with binomial distribution family and logit link function.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# Coefficients and odds ratios
coefficients = pd.DataFrame({
    'feature': X.columns,
    'coef': model.coef_[0],
    'odds_ratio': np.exp(model.coef_[0])
}).sort_values('coef', key=abs, ascending=False)

```

---

### 5. K-Nearest Neighbours (KNN)

KNN predicts using the **nearest examples** in feature space. New observations are classified based on the majority class among their $k$ closest neighbours.

**Key Characteristics:**

* **No model training:** All computation happens at prediction time (lazy learner).
* **Distance-based:** Typically uses Euclidean distance; sensitive to feature scaling.
* **Choice of $k$:** Small $k$ = flexible but noisy; large $k$ = smoother but may miss local patterns.

**Challenges:**

* Slow on large datasets (must compute distances to all training points).
* Sensitive to scaling — always standardise features.
* Curse of dimensionality: performance degrades in high-dimensional spaces.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Always scale features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y_train)

```

---

### 6. Evaluating Classification Models

#### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

#### Key Metrics

| Metric | Formula | Interpretation | When to Focus |
| --- | --- | --- | --- |
| **Accuracy** | $\frac{TP + TN}{\text{Total}}$ | Overall correctness | Balanced classes |
| **Precision** | $\frac{TP}{TP + FP}$ | Of predicted positives, how many are correct? | False positives are costly (e.g., spam filter) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Of actual positives, how many are caught? | False negatives are dangerous (e.g., disease diagnosis) |
| **Specificity** | $\frac{TN}{TN + FP}$ | Of actual negatives, how many are correctly rejected? | Complement to recall for negative class |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean; balances precision and recall | Uneven class distribution; need single metric |

**Critical Warning:** Accuracy is misleading for imbalanced datasets. If fraud occurs in 0.1% of transactions, a model predicting "not fraud" for everything achieves 99.9% accuracy but is completely useless.

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))

```

#### ROC Curve and AUC

The **ROC curve** plots Recall (True Positive Rate) vs. 1 − Specificity (False Positive Rate) across all probability thresholds.

**AUC (Area Under the Curve):**

* 0.50 = Random guessing
* 0.70–0.80 = Reasonable
* 0.80–0.90 = Strong
* 0.90+ = Excellent

**Use:** Comparing models independent of threshold choice. For imbalanced data, also consider the **Precision-Recall curve** (more informative when the positive class is rare).

```python
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

fpr, tpr, _ = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot()

```

#### Lift Chart

Shows improvement over random selection at different percentiles.

$$\text{Lift} = \frac{\text{Response rate in top } p\%}{\text{Overall response rate}}$$

Useful for resource-constrained targeting (e.g., marketing campaigns where you can only contact the top 20% of leads).

---

### 7. The Rare Class Problem and Imbalanced Data

**Challenge:** When class 1 is rare (e.g., fraud = 0.1%), a model predicting all 0s achieves 99.9% accuracy but is useless.

**Solutions:**

| Strategy | Description | When to Use |
| --- | --- | --- |
| **Use Better Metrics** | Precision, recall, AUC, lift (not accuracy) | Always for imbalanced data |
| **Adjust Threshold** | Lower threshold below 0.5 to increase recall | When catching positives is critical |
| **Undersampling** | Randomly remove records from majority class | Abundant majority data; risk losing information |
| **Oversampling** | Duplicate or bootstrap minority class records | Small minority class; risk of overfitting |
| **Class Weighting** | Assign higher weight to minority class in loss function | Use all data while emphasising minority class |
| **SMOTE** | Generate synthetic minority samples by interpolation | Need more diverse minority examples; works best with numeric features |
| **Cost-Sensitive Learning** | Incorporate business costs into decision rule | Misclassification costs are asymmetric and quantifiable |

**SMOTE Algorithm:**

1. For each minority record, find $k$ nearest minority neighbours.
2. Randomly select one neighbour.
3. Create synthetic record: $X_{\text{new}} = X + \lambda \times (X_{\text{neighbour}} - X)$, where $\lambda \sim \text{Uniform}(0, 1)$.
4. Repeat to achieve desired balance.

```python
# Class weighting in logistic regression
model = LogisticRegression(class_weight='balanced', penalty='l2', C=1.0)

# SMOTE oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Threshold optimisation
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    thresholds = np.linspace(0, 1, 101)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx], scores[optimal_idx]

```

---

## Important Formulas

### Naive Bayes

$$P(Y=i \mid X_1, \dots, X_p) \propto P(Y=i) \prod_{j=1}^{p} P(X_j \mid Y=i)$$

### Logistic Regression

$$\text{Logit: } \log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

$$\text{Probability: } p = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_j X_j)}}$$

$$\text{Odds Ratio for } X_j: \; OR = e^{\beta_j}$$

### Classification Metrics

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \frac{TN}{TN + FP}, \quad \text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### LDA Discriminant Function

$$\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log(\pi_k)$$

---

## Common Visualisations

### Confusion Matrix Heatmap

Useful for classification evaluation and error inspection.

```python
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

```

### ROC Curve

Used for threshold comparison and model evaluation.

```python
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_true, y_proba)

```

### Precision-Recall Curve

More informative than ROC when the positive class is rare.

```python
from sklearn.metrics import PrecisionRecallDisplay
PrecisionRecallDisplay.from_predictions(y_true, y_proba)

```

### Decision Boundary Plots

Useful for visualising class separation in 2D feature space.

---

## Machine Learning Relevance

Classification is one of the most widely used ML techniques in business:

* **Customer Churn Prediction:** Predict which customers are likely to leave.
* **Fraud Detection:** Identify fraudulent transactions in real time.
* **Recommendation Systems:** Predict user preference categories.
* **NLP Systems:** Spam detection, sentiment analysis, document classification.
* **Medical Diagnosis:** Disease detection and risk stratification.
* **Credit Scoring:** Predict loan default probability.

---

## Common Pitfalls and Mistakes

1. **Using accuracy with imbalanced data:** A model predicting all negatives can achieve 99% accuracy when positives are 1%. Use precision, recall, and AUC instead.
2. **Ignoring threshold effects:** The default 0.5 threshold is rarely optimal. Optimise based on business costs and precision/recall trade-offs.
3. **Ignoring scaling in KNN:** Distance-based methods require feature standardisation; unscaled features bias distance calculations.
4. **Blindly trusting Naive Bayes assumptions:** Feature independence is often unrealistic; use Naive Bayes for ranking rather than calibrated probabilities.
5. **Misinterpreting probabilities:** A probability of 0.8 does not mean certainty; it means 80% confidence, and 20% of such predictions will be wrong on average.
6. **Data leakage in preprocessing:** Scaling, encoding, and resampling must be fit on training data only, then applied to test data.
7. **Evaluating on training data only:** Always use holdout validation or cross-validation; training metrics are optimistically biased.
8. **Overinterpreting logistic regression coefficients:** Coefficients assume linearity in log-odds; check for nonlinearity with splines or interactions.

---

## Key Takeaways

1. **Classification is fundamentally different from regression:** Output is discrete labels, not continuous values; evaluation requires specialised metrics.
2. **Probability scores enable flexible decision-making:** Don't default to 0.5; optimise thresholds based on business costs and precision/recall trade-offs.
3. **Naive Bayes is fast and scalable:** Despite the independence assumption, it often performs well with text and categorical data.
4. **Logistic regression provides interpretable coefficients:** Odds ratios offer intuitive business interpretation; regularisation helps prevent overfitting.
5. **Imbalanced data requires special handling:** Accuracy is misleading; focus on precision, recall, AUC, and lift. Resampling or weighting can improve minority class detection.
6. **ROC and PR curves tell different stories:** ROC is threshold-invariant and useful for model comparison; PR curves are more informative when the positive class is rare.
7. **No single best classifier:** Performance depends on data characteristics, sample size, feature types, and business objectives. Compare multiple methods via cross-validation.

---

## Connections to Other Chapters

* **Chapter 1:** EDA informs feature selection and reveals class imbalance; boxplots/histograms visualise predictor distributions by class.
* **Chapter 2:** Sampling distributions and bootstrap underpin confidence intervals for classification metrics; binomial distribution models binary outcomes.
* **Chapter 3:** Hypothesis testing concepts (p-values, Type I/II errors) parallel classification error trade-offs; A/B testing applies to model comparison.
* **Chapter 4:** Logistic regression extends linear regression to binary outcomes; similar diagnostics and interpretation principles apply.
* **Chapter 6:** Tree-based methods (random forests, boosting) often outperform the methods in this chapter; ensemble learning builds on single-model foundations.

---

### Questions I Still Have

* When should I prefer logistic regression over tree-based methods for interpretability vs. accuracy?
* How do I properly validate threshold selection to avoid overfitting to validation data?
* What's the most reliable way to handle high-cardinality categorical features in Naive Bayes?
* How can I incorporate business costs directly into model training (not just threshold selection)?
* When is SMOTE appropriate vs. simple oversampling or weighting?

---

## Progress Checklist

* [ ] Read complete chapter (pp. 195–236)
* [ ] Implement Naive Bayes classifier with categorical features
* [ ] Fit LDA and interpret discriminant function weights
* [ ] Fit logistic regression; interpret coefficients as odds ratios
* [ ] Compute confusion matrix, precision, recall, specificity, F1, and AUC
* [ ] Plot ROC and precision-recall curves; compare models
* [ ] Apply undersampling, oversampling, and SMOTE to imbalanced data
* [ ] Optimise classification threshold based on custom cost function
* [ ] Complete `05_classification.ipynb`
* [ ] Solve all exercises in `exercises.ipynb`
* [ ] Experiment with different classifiers in `experiments.ipynb`

---
