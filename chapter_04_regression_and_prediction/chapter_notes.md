# Chapter 04: Regression and Prediction

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Regression is perhaps the most widely used statistical method in data science and one of the most important tools in predictive analytics. At its core, regression models the relationship between one or more predictor variables and a numeric outcome variable.

While classical statistics emphasises explanatory modelling (understanding relationships), data science focuses primarily on predictive modelling (forecasting new outcomes). This chapter bridges both perspectives, showing how regression serves both purposes while highlighting the metrics and practices most relevant to predictive accuracy.

**This chapter introduces:**

- Simple linear regression
- Multiple linear regression
- Prediction and model interpretation
- Factor variables and encoding
- Confounding variables and interaction effects
- Model diagnostics (residuals, outliers, influential values, heteroskedasticity)
- Nonlinear extensions via polynomial and spline terms
- Regularisation concepts

**Regression helps answer questions such as:**
- What factors influence house prices?
- How does advertising affect sales?
- Can we predict customer spending?
- Which variables matter most?

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand regression fundamentals (dependent vs. independent variables)
- Fit and interpret simple and multiple linear regression models
- Interpret regression coefficients and their conditional nature
- Handle factor (categorical) variables via dummy/one-hot encoding
- Evaluate model performance using RMSE, R², and cross-validation
- Diagnose regression problems: outliers, influential values, heteroskedasticity, nonlinearity
- Recognise confounding variables and interaction effects
- Apply polynomial and spline regression for nonlinear relationships
- Use generalised additive models (GAM) for automated nonlinear fitting
- Distinguish prediction intervals from confidence intervals
- Understand the bias-variance trade-off and regularisation basics

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Regression | Modelling relationships between predictors and outcome | `LinearRegression()` |
| Target (Dependent) Variable | Variable being predicted | `y` |
| Predictor (Independent) Variables | Input features used for prediction | `X` |
| Simple Linear Regression | One predictor variable | `LinearRegression()` |
| Multiple Linear Regression | Multiple predictor variables | `LinearRegression()` |
| Coefficient ($\beta$) | Effect size of each predictor | `model.coef_` |
| Intercept ($\beta_0$) | Baseline prediction when all X = 0 | `model.intercept_` |
| Residual | Prediction error (actual − predicted) | `actual - predicted` |
| RMSE | Root Mean Squared Error — average prediction error magnitude | `mean_squared_error()` |
| R² | Coefficient of determination — proportion of variance explained | `model.score()` |
| Adjusted R² | R² penalised for adding non-informative predictors | `statsmodels` summary |
| Confounding Variable | Hidden variable affecting both predictor and outcome | Conceptual |
| Interaction Effect | Effect of one predictor depends on level of another | Feature engineering |
| Multicollinearity | High correlation among predictors | VIF, correlation matrix |
| Overfitting | Model memorises noise rather than learning patterns | Conceptual |
| Regularisation | Penalising model complexity to improve generalisation | `Ridge()`, `Lasso()` |
| Factor Variables | Categorical predictors | `pd.get_dummies()` |
| Splines / GAM | Flexible nonlinear modelling | `patsy`, `pyGAM` |

---

## Key Statistical Ideas

### 1. What Is Regression?

Regression models the relationship between **predictors (X)** and an **outcome (Y)**.

**Goal:** Use known variables to predict unknown outcomes.

**Examples:**
- House size → House price
- Advertising spend → Sales
- Years of experience → Salary

---

### 2. Dependent and Independent Variables

**Dependent Variable (Target, Y):** The variable we want to predict.
- *Examples:* House price, customer churn, airline delay, sales revenue.

**Independent Variables (Features, X):** Variables used to make predictions.
- *Examples:* Income, age, marketing spend, ticket price.

```python
X = df[["income", "age"]]
y = df["spending"]

```

---

### 3. Simple Linear Regression

Uses **one predictor variable** to model the outcome.

**Model Form:**

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:

* $Y$ = response/outcome variable
* $X$ = predictor/feature variable
* $\beta_0$ = intercept (predicted Y when X = 0)
* $\beta_1$ = slope (change in Y per unit change in X)
* $\epsilon$ = error term (unexplained variation)

**Least Squares Estimation:** Minimises the residual sum of squares (RSS):

$$RSS = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2$$

**Coefficient Interpretation:** The coefficient $\beta_1$ tells us how much Y changes when X changes by one unit.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
print(f"Intercept: {model.intercept_}, Coefficients: {model.coef_}")

```

---

### 4. Multiple Linear Regression

Uses **several predictor variables** simultaneously.

**Model Form:**

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

**Coefficient Interpretation (Critical):**
$\beta_j$ = expected change in Y per unit change in $X_j$, **holding all other predictors constant**. This conditional interpretation is crucial — correlated predictors can distort apparent effects.

**Example:** House price prediction may depend on square footage, bedrooms, bathrooms, neighbourhood, and building grade.

```python
X = df[["SqFtTotLiving", "Bedrooms", "Bathrooms", "BldgGrade"]]
y = df["AdjSalePrice"]

model = LinearRegression()
model.fit(X, y)

```

---

### 5. Factor Variables in Regression

**Dummy Variable Encoding (Reference Coding):**

* A factor with $P$ levels creates $P-1$ binary dummy variables.
* One level serves as the reference; coefficients represent differences from that reference.
* *Example:* `PropertyType` with levels {Multiplex, Single Family, Townhouse} → two dummy variables.

**One-Hot Encoding:**

* Creates $P$ binary columns (all levels represented).
* Required for tree-based models, KNN, and neural networks.
* Causes multicollinearity in linear/logistic regression if an intercept is included.

**High-Cardinality Factors (Zip Codes, Product IDs):**

* Strategies include grouping by outcome (median residual by zip code), target encoding, or embeddings.

```python
# Reference coding (for linear regression)
X_ref = pd.get_dummies(df[['PropertyType', 'ZipCode']], drop_first=True)

# One-hot encoding (for tree models, KNN)
X_encoded = pd.get_dummies(df[['PropertyType', 'ZipCode']], drop_first=False)

```

---

### 6. Interpreting the Regression Equation

**Correlated Predictors:** When $X_1$ and $X_2$ are correlated, coefficients become unstable and hard to interpret. For example, the `Bedrooms` coefficient may appear negative in a housing model because larger homes (which have more bedrooms) are already captured by `SqFtTotLiving`.

**Multicollinearity:** Perfect or near-perfect correlation among predictors causes large standard errors and unstable coefficients. Detected via Variance Inflation Factor (VIF) > 5–10. Solutions: remove redundant variables, combine correlated predictors, or use PCA.

**Confounding Variables:** An omitted variable that affects both predictor and outcome biases coefficient estimates. *Example:* Location omitted from housing model biases `Bathrooms` coefficient. Solution: include important confounders or use causal inference methods.

**Interaction Effects:** The effect of $X_1$ on $Y$ depends on the level of $X_2$.

**Model with interaction:**


$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \times X_2) + \epsilon$$

```python
df["interaction"] = df["ad_spend"] * df["age"]

```

---

### 7. Model Performance Metrics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (Y_i - \hat{Y}_i)^2}$ | Average prediction error (same units as Y). Lower is better. |
| **RSE** | $\sqrt{\frac{RSS}{n-p-1}}$ | RMSE adjusted for degrees of freedom. |
| **R²** | $1 - \frac{RSS}{TSS}$ | Proportion of variance explained (0 to 1). 0 = no explanatory power; 1 = perfect prediction. |
| **Adjusted R²** | $1 - (1-R^2)\frac{n-1}{n-p-1}$ | Penalises adding non-informative predictors. |

**Important:** High R² does NOT guarantee good generalisation. Always validate on held-out data.

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

```

---

### 8. Residuals and Diagnostics

**Residuals** are prediction errors:


$$\text{Residual} = \text{Actual} - \text{Predicted}$$

Good regression models have small, randomly scattered residuals. Residual analysis helps identify poor fit, missing relationships, outliers, and assumption violations.

**Key Diagnostic Checks:**

| Issue | Detection | Fix |
| --- | --- | --- |
| **Outliers** | Standardised residual > 2–3 | Investigate, robust regression |
| **Influential Values** | Cook's distance > $4/(n-p-1)$ | Investigate, robust methods |
| **Heteroskedasticity** | Funnel shape in residuals vs. fitted plot | Transform Y, weighted regression, robust SE |
| **Non-Normal Residuals** | QQ-plot deviation from diagonal | Transform Y, bootstrap inference |
| **Nonlinearity** | Partial residual plot shows curves | Polynomial terms, splines, GAM |

```python
# Residual diagnostic plots
residuals = model_sm.resid
fitted = model_sm.fittedvalues

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Residuals vs fitted
axes[0].scatter(fitted, residuals, alpha=0.5)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# QQ-plot
sm.qqplot(residuals, line='45', ax=axes[1])
axes[1].set_title('Normal Q-Q')

# Scale-Location
std_resid = np.sqrt(np.abs(residuals / residuals.std()))
axes[2].scatter(fitted, std_resid, alpha=0.5)
axes[2].set_xlabel('Fitted values')
axes[2].set_ylabel('√|Standardized Residuals|')
axes[2].set_title('Scale-Location')

plt.tight_layout()
plt.show()

```

---

### 9. Polynomial and Spline Regression

**Polynomial Regression:** Add powers of predictor:


$$Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon$$

* Simple and interpretable for low degrees.
* Can be unstable and "wiggly" at boundaries; poor for extrapolation.

**Spline Regression:** Piecewise polynomials joined smoothly at knots.

* **Cubic splines:** Continuous first and second derivatives.
* **Natural splines:** Linear beyond boundary knots (better extrapolation).
* Knot selection: quantiles of predictor, domain knowledge, or automated via GAM.

**Generalised Additive Models (GAM):**


$$Y = \beta_0 + f_1(X_1) + f_2(X_2) + \cdots + f_p(X_p) + \epsilon$$

* Each $f_j$ is a smooth function (typically a spline) estimated from data.
* Advantages: automatic knot selection, interpretable smooths, handles nonlinearity automatically.

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['SqFtTotLiving']])

# GAM with pyGAM
from pygam import LinearGAM, s, l
gam = LinearGAM(s(0) + l(1) + l(2) + l(3)).fit(X, y)
gam.summary()

```

---

### 10. Prediction Intervals vs. Confidence Intervals

| Interval Type | Purpose | Width |
| --- | --- | --- |
| **Confidence Interval** | Uncertainty in the *mean* prediction | Narrower |
| **Prediction Interval** | Uncertainty in an *individual* prediction | Wider (includes individual variation $\sigma^2$) |

**Extrapolation Danger:** Predicting outside the range of training data yields unreliable results. Always document predictor ranges for safe deployment.

---

### 11. Overfitting and the Bias-Variance Trade-Off

**Overfitting:** The model memorises noise instead of learning patterns. Symptoms: excellent training performance, poor test performance. Causes: too many variables, small datasets, overly complex models.

**Bias-Variance Trade-Off:**

* **High bias:** Model too simple, underfitting.
* **High variance:** Model too complex, overfitting.
* **Goal:** Balanced generalisation.

**Cross-Validation for Prediction:** Split data into training/validation folds, evaluate on held-out data, repeat, and average results to estimate out-of-sample performance.

```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"CV RMSE: {-cv_scores.mean():,.0f} ± {-cv_scores.std():,.0f}")

```

---

### 12. Regularisation

Regularisation reduces model complexity by penalising large coefficients.

**Ridge Regression (L2):** Shrinks coefficients toward zero but never exactly to zero. Useful for multicollinearity.

**Lasso Regression (L1):** Can force coefficients exactly to zero, performing automatic feature selection.

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

```

---

## Important Formulas

### Simple Linear Regression

$$Y = \beta_0 + \beta_1 X + \epsilon$$

$$\hat{\beta}_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}, \quad \hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$

### Multiple Linear Regression

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$$

### Model Evaluation

$$RMSE = \sqrt{\frac{1}{n}\sum (Y_i - \hat{Y}_i)^2}, \quad R^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2}$$

$$\text{Adjusted } R^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}$$

### Diagnostics

$$\text{Standardized Residual: } r_i = \frac{e_i}{s\sqrt{1 - h_i}}$$

$$\text{Cook's Distance: } D_i = \frac{r_i^2}{p} \times \frac{h_i}{1 - h_i}$$

### Nonlinear Extensions

$$\text{Polynomial: } Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_k X^k + \epsilon$$

$$\text{Cubic Spline: } f(X) = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \sum \gamma_j (X - \kappa_j)_+^3$$

---

## Common Visualisations

### Scatterplot with Regression Line

Used to inspect relationships, trends, and linearity.

```python
sns.regplot(data=df, x="X", y="Y")

```

### Residual Diagnostic Plots

Used to inspect prediction quality and model assumptions. Good residuals are randomly scattered with constant variance.

### Correlation Heatmap

Useful for detecting multicollinearity among predictors.

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)

```

### Partial Residual Plots

Visualise the relationship between one predictor and the outcome, adjusting for all other predictors. Used to detect nonlinearity.

---

## Machine Learning Relevance

Regression is central to machine learning.

**Predictive Modelling:** Used in demand forecasting, revenue prediction, delay prediction, and pricing systems.

**Feature Importance:** Regression coefficients help understand which variables matter most (with appropriate scaling).

**Baseline Models:** Linear regression is often the first benchmark before trying more complex ML models.

**Regularisation:** Lasso and Ridge are widely used to prevent overfitting, improve generalisation, and perform feature selection.

**Handling Nonlinearity:** Splines and GAMs bridge the gap between interpretable linear models and flexible black-box models.

---

## Common Pitfalls and Mistakes

1. **Assuming correlation implies causation:** Regression identifies association, not proof of causality.
2. **Ignoring multicollinearity:** Highly correlated features destabilise coefficients. Check VIF or correlation matrix.
3. **Overfitting:** Too many features often reduce generalisation. Use regularisation or feature selection.
4. **Ignoring residual analysis:** Residuals reveal bad assumptions, missing patterns, and poor model fit.
5. **Evaluating only on training data:** Always validate performance using test or cross-validation sets.
6. **Extrapolating beyond data range:** Predictions outside training predictor ranges are unreliable.
7. **Confusing confidence and prediction intervals:** Using confidence intervals for individual predictions underestimates uncertainty.
8. **Treating categorical variables as numeric:** Encoding zip codes or categories as integers imposes false ordering.
9. **Omitting confounders:** Biases coefficient estimates; include domain-relevant variables or use causal methods.
10. **Relying solely on R²:** High R² doesn't guarantee good predictions; always check out-of-sample performance.

---

## Key Takeaways

1. **Regression is flexible but assumptions matter:** Linearity, independence, homoskedasticity, and normality (for inference) should be checked — but prediction may tolerate violations.
2. **Interpret coefficients conditionally:** "Holding other predictors constant" is critical; correlated predictors can reverse or mask true effects.
3. **Factor variables require careful encoding:** Reference coding for linear models; one-hot for tree-based models; high-cardinality factors need special handling.
4. **Diagnostics are essential:** Residual plots, influence measures, and partial residual plots reveal model misspecification, outliers, and nonlinearity.
5. **Nonlinearity is common:** Polynomial terms offer simple extensions; splines and GAMs provide flexible, automated nonlinear fitting.
6. **Prediction ≠ explanation:** Focus on out-of-sample metrics (RMSE, cross-validation) for prediction; R² and p-values matter more for explanation.
7. **Cross-validation prevents overfitting:** Always evaluate models on held-out data; use k-fold CV for robust performance estimates.
8. **Avoid extrapolation:** Predictions outside the training data range are unreliable; document predictor ranges for safe deployment.

---

## Connections to Other Chapters

* **Chapter 1:** EDA tools (scatterplots, boxplots) inform predictor selection and reveal nonlinearities before modelling.
* **Chapter 2:** Sampling distributions and bootstrap underpin confidence intervals for regression coefficients.
* **Chapter 3:** t-tests and F-tests for coefficient significance; permutation tests as assumption-free alternatives.
* **Chapter 5:** Logistic regression extends linear regression to binary outcomes; similar diagnostics and interpretation principles apply.
* **Chapter 6:** Tree-based models (random forests, boosting) automatically handle nonlinearities and interactions that regression requires manual specification for.
* **Chapter 7:** PCA can reduce dimensionality before regression; clustering can identify subpopulations for stratified modelling.

---

### Questions I Still Have

* How do I decide between polynomial, spline, and GAM for a given nonlinear relationship?
* When is it appropriate to use reference coding vs. one-hot encoding in practice?
* What's the best way to handle high-cardinality factors like zip codes without overfitting?
* How can I automate residual diagnostic checks in a production pipeline?
* When should I prioritise interpretability (linear model) vs. predictive accuracy (ensemble methods)?

---

## Progress Checklist

* [ ] Read complete chapter (pp. 141–194)
* [ ] Fit simple and multiple linear regression models
* [ ] Implement dummy variable encoding for factor variables
* [ ] Create residual diagnostic plots (residuals vs. fitted, QQ, scale-location)
* [ ] Detect and investigate outliers/influential points using Cook's distance
* [ ] Fit polynomial and spline models to capture nonlinearity
* [ ] Apply cross-validation to estimate out-of-sample RMSE
* [ ] Complete `04_regression_and_prediction.ipynb`
* [ ] Solve all exercises in `exercises.ipynb`
* [ ] Experiment with GAM and interaction terms in `experiments.ipynb`

---
