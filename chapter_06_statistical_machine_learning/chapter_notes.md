# Chapter 06: Statistical Machine Learning

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Statistical machine learning represents a shift from classical statistical methods—which impose linear or parametric structure on data—to flexible, data-driven approaches that learn patterns directly from observations. It combines statistical reasoning with computational methods to identify patterns and make predictions from data.

Unlike traditional statistical modelling, where interpretation often matters most, machine learning focuses heavily on:

- Prediction accuracy
- Generalisation to unseen data
- Handling high-dimensional data
- Automated pattern discovery

The core idea: instead of fitting a single global model, we combine multiple models (an ensemble) or let the data speak through local, adaptive rules. K-Nearest Neighbours provides an intuitive baseline; tree models discover hierarchical decision rules; bagging and boosting aggregate many trees to dramatically improve predictive accuracy while managing overfitting.

**The central question becomes:** *How can we make better predictions from data?*

Statistical machine learning sits at the intersection of statistics, computer science, and optimisation, forming the foundation of modern AI systems.

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand statistical machine learning fundamentals and distinguish prediction from inference
- Apply K-Nearest Neighbours (KNN) for classification and regression
- Build and interpret decision tree models using recursive partitioning
- Measure impurity using Gini index and entropy
- Control tree complexity to avoid overfitting (pruning, hyperparameters)
- Implement bagging and random forests to improve tree stability
- Apply boosting algorithms (AdaBoost, gradient boosting, XGBoost)
- Use regularisation and cross-validation to tune ensemble models
- Extract and interpret variable importance from ensemble models
- Recognise overfitting and underfitting; understand the bias-variance trade-off
- Distinguish when to use single trees vs. ensembles for interpretability vs. accuracy

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Machine Learning | Pattern learning from data | `scikit-learn`, `xgboost` |
| Prediction | Estimating unknown outcomes | `predict()`, `predict_proba()` |
| K-Nearest Neighbours (KNN) | Instance-based learning using similar records | `KNeighborsClassifier()`, `KNeighborsRegressor()` |
| Decision Tree | Rule-based prediction via recursive partitioning | `DecisionTreeClassifier()`, `DecisionTreeRegressor()` |
| Gini Impurity | Measure of node purity; $p(1-p)$ for binary case | `criterion='gini'` |
| Entropy | Alternative impurity measure; more sensitive to probability changes | `criterion='entropy'` |
| Bagging | Bootstrap aggregation — train models on bootstrap samples | `BaggingClassifier()`, `BaggingRegressor()` |
| Random Forest | Ensemble of trees with bootstrap samples + random feature subsets | `RandomForestClassifier()`, `RandomForestRegressor()` |
| Boosting | Sequential learning; each model corrects predecessors' errors | `GradientBoostingClassifier()`, `XGBClassifier()` |
| XGBoost | Optimised gradient boosting with regularisation | `xgb.XGBClassifier()`, `xgb.XGBRegressor()` |
| Ensemble Learning | Combining multiple models for better performance | Multiple models combined |
| Variable Importance | Ranking predictors by contribution to predictions | `feature_importances_`, permutation importance |
| Training Set | Data used for learning | `train_test_split()` |
| Test Set | Held-out data for evaluation | Holdout data |
| Cross-Validation | Robust evaluation via repeated train-test splits | `cross_val_score()`, `GridSearchCV()` |
| Overfitting | Memorising noise; excellent training, poor test performance | Conceptual |
| Underfitting | Model too simple; poor performance everywhere | Conceptual |
| Bias-Variance Trade-Off | Balance between model simplicity and flexibility | Conceptual |

---

## Key Statistical Ideas

### 1. Statistical Machine Learning Philosophy

Machine learning focuses on **prediction quality** rather than only **model interpretation**.

| | Traditional Statistics | Machine Learning |
|---|---|---|
| **Primary Question** | Why does this happen? | Can we predict accurately? |
| **Goal** | Inference, explanation | Prediction, generalisation |
| **Model Complexity** | Parsimonious, interpretable | Flexible, may be complex |
| **Examples** | Linear regression, logistic regression | Random forests, XGBoost, neural networks |

**Examples of ML prediction tasks:**
- Fraud detection
- Customer churn prediction
- Flight delay prediction
- Product recommendation

---

### 2. Training and Test Sets

Data is split to prevent overfitting and evaluate generalisation.

**Training Set:** Used to train the model.
**Test Set:** Used to evaluate performance on unseen data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

```

**Purpose:** The test set simulates how the model will perform on new, unseen data. Never use test data during model training or hyperparameter tuning.

---

### 3. K-Nearest Neighbours (KNN)

KNN predicts based on the **K most similar nearby observations** in feature space.

**Algorithm:**

1. For each new record, compute distance to all training records.
2. Select the K nearest neighbours.
3. **Classification:** Assign majority class among neighbours.
4. **Regression:** Assign average value of neighbours.

**Distance Metrics:**

| Metric | Formula | Use Case |
| --- | --- | --- |
| **Euclidean** | $\sqrt{\sum (x_i - u_i)^2}$ | Default for continuous variables |
| **Manhattan** | $\sum | x_i - u_i |
| **Mahalanobis** | Accounts for covariance | When predictors are correlated |

**Critical Considerations:**

* **Standardisation is essential:** Variables on large scales dominate distance calculations. Always scale features.
* **Choice of K:** Small K → high variance (overfitting); large K → high bias (oversmoothing). Use cross-validation to select K.
* **Curse of dimensionality:** Distance metrics become less meaningful in high-dimensional spaces.

**KNN as a Feature Engine:** Run KNN to generate local propensity scores, then add these scores as features to a second-stage model — combines local knowledge with global modelling power.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y_train)

```

**Advantages:** Simple, intuitive, no training phase.
**Limitations:** Slow at scale, sensitive to scaling, struggles with many features.

---

### 4. Decision Trees

Decision trees create **if-then decision rules** through recursive partitioning of the predictor space.

**Recursive Partitioning Algorithm:**

1. Start with all data in one node.
2. For each predictor and each possible split value, measure impurity reduction.
3. Choose the split that maximises impurity reduction.
4. Repeat recursively on each subset until stopping criteria are met.

**Impurity Measures (Binary Case):**

| Measure | Formula | Properties |
| --- | --- | --- |
| **Gini Impurity** | $p(1-p)$ | Computationally efficient; default in many implementations |
| **Entropy** | $-p\log_2(p) - (1-p)\log_2(1-p)$ | More sensitive to probability changes |
| **Classification Error** | $\min(p, 1-p)$ | Less sensitive; rarely used for splitting |

**Controlling Tree Growth (Preventing Overfitting):**

* `max_depth`: Maximum tree depth.
* `min_samples_split`: Minimum records required to split a node.
* `min_samples_leaf`: Minimum records required in a terminal node.
* `min_impurity_decrease`: Minimum impurity reduction required for a split.
* **Pruning:** Post-hoc removal of branches that don't improve validation performance.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
tree.fit(X_train, y_train)

# Visualise
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, filled=True)
plt.show()

```

**Advantages:** Interpretable decision rules, handles nonlinear relationships and interactions automatically, robust to outliers, no distributional assumptions.
**Limitations:** High variance (small data changes can produce very different trees), axis-aligned splits may miss diagonal boundaries.

---

### 5. Overfitting and Underfitting

**Overfitting:** The model memorises noise instead of learning general patterns.

* *Symptoms:* Excellent training performance, poor test performance.
* *Example:* Very deep decision trees that perfectly fit training data but fail on new data.

**Underfitting:** The model is too simple to learn meaningful patterns.

* *Symptoms:* Poor performance on both training and test data.
* *Example:* A tree with depth = 1 for a complex problem.

---

### 6. Bias-Variance Trade-Off

Model complexity creates a fundamental trade-off:

|  | High Bias | High Variance |
| --- | --- | --- |
| **Description** | Too simple; misses patterns | Too complex; memorises noise |
| **Problem** | Underfitting | Overfitting |
| **Example** | Linear model for nonlinear data | Overgrown decision tree |

**Goal:** Find the optimal balance that minimises total prediction error.

Conceptually: $\text{Prediction Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$

---

### 7. Bagging (Bootstrap Aggregation)

Bagging reduces variance by training many models on bootstrap samples and combining their predictions.

**Algorithm:**

1. Draw $B$ bootstrap samples (with replacement) from training data.
2. Fit a model (typically a tree) to each bootstrap sample.
3. **Classification:** Majority vote across all models.
4. **Regression:** Average predictions across all models.

```python
from sklearn.ensemble import BaggingClassifier

```

**Benefits:** More stable predictions, reduced overfitting compared to single trees.

---

### 8. Random Forests

Random forests improve upon bagging by decorrelating trees through random feature selection.

**Key Innovation:** At each split, consider only a random subset of features — typically $\sqrt{p}$ for classification, $p/3$ for regression.

**Algorithm:**

1. Draw $B$ bootstrap samples from training data.
2. For each sample, grow a tree. At each node, randomly select $m$ predictors and find the best split among them.
3. Combine predictions via majority vote (classification) or averaging (regression).

**Out-of-Bag (OOB) Error:**

* Each bootstrap sample leaves out ~37% of records.
* Use these "out-of-bag" records to estimate prediction error without a separate validation set.

**Variable Importance:**

| Method | Description | Reliability |
| --- | --- | --- |
| **Mean Decrease in Accuracy** | Permute each variable; measure drop in OOB accuracy | High (uses out-of-sample data) |
| **Mean Decrease in Impurity (Gini)** | Sum of impurity reductions across all splits | Moderate (training data only) |

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features='sqrt',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Error: {1 - rf.oob_score_:.3f}")

```

**Advantages:** Strong performance, robust, handles nonlinear relationships, resistant to overfitting, provides variable importance.
**Limitations:** Less interpretable than single trees; can be computationally intensive.

---

### 9. Boosting

Boosting learns **sequentially** — each new model focuses on records that previous models mispredicted.

**AdaBoost Algorithm:**

1. Initialise equal weights for all training records.
2. For each iteration $m = 1$ to $M$:
* Fit a weak learner (shallow tree) to weighted data.
* Compute weighted error $e_m$.
* Compute model weight: $\alpha_m = \log((1-e_m)/e_m)$.
* Increase weights for misclassified records.


3. Final prediction: weighted vote of all $M$ models.

**Gradient Boosting:**

* Frames boosting as gradient descent on a loss function.
* Each new model fits the negative gradient (pseudo-residuals) of the loss.
* More flexible than AdaBoost; supports various loss functions.

**Stochastic Gradient Boosting:**

* Adds randomness: subsample records and/or predictors at each iteration.
* Reduces overfitting, improves generalisation.

**XGBoost Key Features:**

* Regularisation (L1/L2) on tree complexity to prevent overfitting.
* Efficient handling of sparse data and missing values.
* Parallel processing for faster training.
* Built-in cross-validation and early stopping.

**Critical Hyperparameters for Boosting:**

| Parameter | Effect | Typical Range |
| --- | --- | --- |
| `n_estimators` | Number of boosting rounds | 100–1000 |
| `learning_rate` (eta) | Shrinkage factor for each tree | 0.01–0.3 |
| `max_depth` | Maximum tree depth (shallow trees typical) | 3–8 |
| `subsample` | Fraction of records to sample per round | 0.5–1.0 |
| `colsample_bytree` | Fraction of predictors to sample per tree | 0.5–1.0 |
| `reg_alpha` / `reg_lambda` | L1/L2 regularisation strength | 0–10 |

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    reg_lambda=1.0,
    random_state=42
)

```

**Advantages:** Highly accurate, strong predictive power.
**Limitations:** Slower than random forests, sensitive to noise, requires careful hyperparameter tuning.

---

### 10. Ensemble Learning

Ensemble learning combines multiple models to improve performance beyond any single model.

**Core Idea:** Many weak learners combined form a stronger learner.

**Examples:** Random forests (parallel ensemble), boosting (sequential ensemble).

Often performs better than any individual model.

---

### 11. Cross-Validation

Cross-validation evaluates model performance robustly by repeatedly splitting data.

**K-Fold Cross-Validation:**

1. Split data into $K$ equal folds.
2. For each fold: train on $K-1$ folds, evaluate on the held-out fold.
3. Average performance across all $K$ iterations.

**Benefits:** More stable performance estimates, less random variation than a single train-test split.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

```

**For ensemble tuning:** OOB error provides efficient validation for random forests; explicit cross-validation is needed for boosting hyperparameter tuning.

---

## Important Formulas

### Distance Metrics

$$\text{Euclidean: } d(x, u) = \sqrt{\sum (x_i - u_i)^2}$$

$$\text{Manhattan: } d(x, u) = \sum |x_i - u_i|$$

### Impurity Measures (Binary Case)

$$\text{Gini: } I(A) = p(1-p)$$

$$\text{Entropy: } I(A) = -p\log_2(p) - (1-p)\log_2(1-p)$$

### Boosting Weight Updates (AdaBoost)

$$\alpha_m = \log\left(\frac{1 - e_m}{e_m}\right)$$

$$w_i^{\text{(new)}} = w_i^{\text{(old)}} \times \exp(\alpha_m \times \mathbb{I}(\text{misclassified}))$$

### Random Forest Prediction

$$\text{Classification: } f(x) = \text{majority\_vote}\{f_1(x), f_2(x), \dots, f_B(x)\}$$

$$\text{Regression: } f(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

### XGBoost Regularised Objective

$$\text{Obj} = \sum L(y_i, \hat{y}_i) + \sum \left[ \gamma T + \frac{1}{2}\lambda ||w||^2 + \alpha ||w||_1 \right]$$


Where $T$ = number of leaves, $w$ = leaf weights.

---

## Common Visualisations

### Decision Tree Plots

Useful for understanding rule structure and model interpretability.

```python
from sklearn.tree import plot_tree
plot_tree(model, feature_names=X.columns, filled=True)

```

### Feature Importance Plots

Used to inspect which predictors matter most in random forests and boosting models.

```python
# sklearn random forest
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=importances.head(15), x='importance', y='feature')

```

### Cross-Validation Score Plots

Used for comparing models and assessing performance stability across folds.

### Error Curves (Training vs. Validation)

Useful for detecting overfitting and guiding hyperparameter tuning. Plot training and validation error as a function of model complexity or number of iterations.

---

## Machine Learning Relevance

This chapter is foundational for modern ML applications:

* **Recommendation Systems:** Predict user preferences and product affinity.
* **Fraud Detection:** Detect suspicious behaviour patterns in real time.
* **Airline Operations:** Predict flight delays, demand, and disruptions.
* **Customer Analytics:** Predict churn, engagement, and purchase likelihood.
* **Feature Importance:** Tree models help answer: *What matters most for predictions?*

---

## Common Pitfalls and Mistakes

1. **Evaluating only training performance:** Always evaluate on unseen test data; training metrics are optimistically biased.
2. **Overfitting decision trees:** Deep trees memorise noise; use `max_depth`, `min_samples_leaf`, or pruning.
3. **Ignoring scaling in KNN:** Distance-based methods require standardisation; unscaled features bias distance calculations.
4. **Choosing K arbitrarily in KNN:** Always use cross-validation; odd K avoids ties in binary classification.
5. **Using one train-test split only:** Prefer cross-validation for robust, stable evaluation.
6. **Ignoring class imbalance in ensembles:** Use `class_weight`, sampling strategies, or evaluation metrics beyond accuracy.
7. **Tuning boosting without early stopping:** Can lead to severe overfitting; monitor validation loss during training.
8. **Misinterpreting Gini importance:** Training-data-based; prefer permutation importance for reliable rankings.
9. **Using ensembles when interpretability is required:** Consider single trees or logistic regression if stakeholders need transparent rules.

---

## Key Takeaways

1. **KNN is simple but sensitive to scaling:** Always standardise features; choose K via cross-validation; beware of high-dimensional spaces.
2. **Single trees are interpretable but unstable:** Use for exploration and rule extraction; ensemble methods for production prediction.
3. **Random forests reduce variance through averaging:** Bootstrap sampling + random feature selection decorrelates trees, producing more robust predictions.
4. **Boosting reduces bias through sequential learning:** Each model corrects predecessors' errors; requires careful tuning to avoid overfitting.
5. **XGBoost is powerful but complex:** Start with defaults, then tune `learning_rate` and `n_estimators` together; use early stopping.
6. **Variable importance guides feature selection:** Permutation-based importance is more reliable than Gini-based; interpret in context.
7. **Cross-validation is non-negotiable:** OOB error helps for random forests; explicit CV needed for boosting hyperparameter tuning.
8. **Interpretability vs. accuracy trade-off:** Single trees offer transparent rules; ensembles offer superior accuracy but require SHAP/LIME for interpretation.

---

## Connections to Other Chapters

* **Chapter 1:** EDA informs feature selection for KNN/trees; boxplots reveal outliers that affect distance metrics.
* **Chapter 2:** Bootstrap sampling underpins bagging; understanding sampling distributions helps interpret ensemble variability.
* **Chapter 3:** Permutation tests parallel permutation importance; A/B testing frameworks apply to model comparison.
* **Chapter 4:** Tree models automatically capture interactions that regression requires manual specification for.
* **Chapter 5:** KNN and trees are alternative classifiers to logistic regression; ensemble methods often outperform single-model approaches.
* **Chapter 7:** PCA can reduce dimensionality before KNN; clustering can identify subpopulations for stratified tree modelling.

---

## Progress Checklist

* [ ] Read complete chapter (pp. 237–282)
* [ ] Implement KNN with standardisation and K selection via cross-validation
* [ ] Build and visualise a decision tree; interpret splitting rules
* [ ] Train random forest; extract and plot variable importance
* [ ] Implement XGBoost with hyperparameter tuning via `GridSearchCV`
* [ ] Compare model performance (accuracy, AUC, training time) across methods
* [ ] Apply permutation importance to assess feature relevance
* [ ] Complete `06_statistical_machine_learning.ipynb`
* [ ] Solve all exercises in `exercises.ipynb`
* [ ] Experiment with different hyperparameters in `experiments.ipynb`

---

### Questions I Still Have

* How do I decide between random forest and XGBoost for a given problem? (RF for robustness, XGB for marginal accuracy gains with careful tuning)
* When is KNN preferable to tree-based methods? (Low-dimensional, smooth decision boundaries, need for propensity scores)
* How can I make ensemble models more interpretable for stakeholders? (SHAP values, partial dependence plots, rule extraction from single trees)
* What's the most efficient workflow for hyperparameter tuning in XGBoost? (Start with learning_rate + n_estimators, then max_depth, then regularisation)
* How do I handle categorical variables in tree ensembles without one-hot encoding blowup?

---
