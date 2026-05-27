# Chapter 01: Exploratory Data Analysis

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck  
> **Historical Context:** EDA was pioneered by John Tukey in 1977, who emphasised understanding data through summary statistics and visualisations before formal modelling.

---

## Chapter Overview

Exploratory Data Analysis (EDA) is the process of examining, summarising, and visualising data before applying statistical models or machine learning algorithms. It is the first and most crucial step in any data science project.

**Primary goals of EDA:**

- Understand the structure and shape of the data
- Identify patterns, trends, and anomalies
- Detect missing values and outliers
- Understand distributions and relationships between variables
- Generate hypotheses for further analysis

EDA is foundational because poor understanding of data almost always leads to poor modelling decisions.

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand the role and importance of EDA in the data science workflow
- Distinguish between data types (numeric, categorical, binary, ordinal)
- Work with rectangular data structures (data frames)
- Calculate and interpret measures of central tendency (mean, median, trimmed mean, weighted mean)
- Understand variability and dispersion (variance, standard deviation, IQR, MAD)
- Work with percentiles, quartiles, and boxplots
- Identify and investigate outliers
- Visualise data distributions using histograms, density plots, and boxplots
- Interpret correlations between variables
- Explore relationships between multiple variables
- Perform basic exploratory analysis using Python

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Mean | Arithmetic average of values | `df.mean()` |
| Median | Middle value in sorted data (robust to outliers) | `df.median()` |
| Trimmed Mean | Mean after dropping extreme values from both ends | `scipy.stats.trim_mean()` |
| Weighted Mean | Mean where observations carry different importance | `np.average()` |
| Percentiles | Value below which a given percentage of observations fall | `np.percentile()` / `df.quantile()` |
| Variance | Spread of data around the mean | `df.var()` |
| Standard Deviation | Average variability from the mean (original units) | `df.std()` |
| IQR | Spread of the middle 50% of data (Q3 − Q1) | `df.quantile(0.75) - df.quantile(0.25)` |
| Median Absolute Deviation (MAD) | Robust measure of spread | `scipy.stats.median_abs_deviation()` |
| Boxplot | Visual summary of distribution + outlier detection | `sns.boxplot()` |
| Frequency Table | Count of categorical values | `df['col'].value_counts()` |
| Histogram | Distribution of numerical data using bins | `plt.hist()` / `sns.histplot()` |
| Density Plot | Smoothed version of histogram (KDE) | `sns.kdeplot()` |
| Pearson Correlation | Strength of linear relationship (−1 to +1) | `df.corr(numeric_only=True)` |
| Scatterplot | Relationship between two numerical variables | `sns.scatterplot()` |
| Heatmap | Visual matrix of correlations | `sns.heatmap()` |

---

## Key Statistical Ideas

### 1. Elements of Structured Data

**Data Types:**

- **Numeric Data**
  - *Continuous:* Can take any value in an interval (e.g., temperature, time)
  - *Discrete:* Integer values, counts (e.g., number of customers)

- **Categorical Data**
  - Takes a fixed set of values (e.g., state names, product categories)
  - *Binary:* Special case with two values (0/1, yes/no, true/false)
  - *Ordinal:* Categories with explicit ordering (e.g., ratings 1–5)

**Why Data Types Matter:**
- Determines appropriate visualisations
- Influences statistical model selection
- Affects computational performance in software

### 2. Rectangular Data

**Structure:**
- **Rows** = Records (cases, observations, instances)
- **Columns** = Features (variables, predictors, attributes)
- **Data Frame** = The standard format in Python (pandas) and R

**Key Terminology:**
- **Feature / Predictor:** Input variable used for prediction
- **Outcome / Target:** Variable being predicted
- **Index:** Row identifier for efficient queries

---

### 3. Estimates of Location (Central Tendency)

Measures of location describe the centre of the data.

| Metric | Formula | When to Use | Robust to Outliers? |
|---|---|---|---|
| **Mean** | \(\bar{x} = \frac{\sum x_i}{n}\) | Symmetric distributions | ❌ No |
| **Median** | Middle value of sorted data | Skewed distributions | ✅ Yes |
| **Trimmed Mean** | Mean after dropping p% from each end | Compromise between mean and median | ✅ Yes |
| **Weighted Mean** | \(\bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}\) | Observations have different importance | ❌ No |

**Key Insight from the Book:**
For state population data: Mean (6.16M) > Trimmed Mean (4.78M) > Median (4.44M).  
This pattern indicates right-skewed data with high outliers pulling the mean upward.

**Python Examples:**
```python
df["column"].mean()                     # Mean
df["column"].median()                   # Median
trim_mean(df["column"], 0.1)            # 10% trimmed mean
np.average(df["column"], weights=...)   # Weighted mean
```

---

### 4. Estimates of Variability (Dispersion)

Measures of variability describe how spread out the data is.

| Metric | Formula | Interpretation | Robust? |
|---|---|---|---|
| **Variance** | \(s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}\) | Average squared deviation from mean | ❌ |
| **Standard Deviation** | \(s = \sqrt{s^2}\) | Typical deviation from mean (original units) | ❌ |
| **Interquartile Range (IQR)** | \(Q3 - Q1\) | Spread of middle 50% | ✅ |
| **Median Absolute Deviation (MAD)** | \(\text{median}(|x_i - \text{median}(x)|)\) | Robust measure of spread | ✅ |
| **Range** | \(\max - \min\) | Total spread | ❌ |

**Key Points:**
- Use \(n-1\) (degrees of freedom) for unbiased population estimates
- Standard deviation is always ≥ Mean Absolute Deviation ≥ MAD
- For skewed distributions, **prefer IQR or MAD** over standard deviation
- Standard deviation is critical for z-scores, confidence intervals, and ML preprocessing

**Python Examples:**
```python
df["column"].var()                      # Variance
df["column"].std()                      # Standard deviation
df["column"].quantile(0.75) - df["column"].quantile(0.25)  # IQR
scipy.stats.median_abs_deviation(df["column"])              # MAD
```

---

### 5. Exploring Data Distributions

Understanding distributions helps identify skewness, symmetry, multimodality, and unusual observations.

#### Boxplot
Shows quartiles (25th, 50th, 75th percentiles) with whiskers extending to 1.5 × IQR. Points beyond whiskers are flagged as potential outliers.
```python
sns.boxplot(x=df["column"])
```

#### Histogram
Divides data into equal-width bins to show frequency. Bin width critically affects interpretation—empty bins should be included.
```python
sns.histplot(df["column"], bins=30)
```

#### Density Plot
Smoothed version of a histogram using kernel density estimation. The area under the curve equals 1. Better for comparing multiple distributions.
```python
sns.kdeplot(data=df, x="column")
```

**Percentiles:**
- The p-th percentile is the value below which p% of observations fall
- Quartiles: 25th (Q1), 50th (Median), 75th (Q3)
- Deciles: 10th, 20th, …, 90th percentiles

---

### 6. Exploring Binary and Categorical Data

**Summary Statistics:**
- **Mode:** Most frequently occurring category
- **Expected Value:** \(\sum (\text{value} \times \text{probability})\)
- **Proportions / Percentages:** For each category

**Visualisation:**
- **Bar Charts:** Categories are distinct and bars are separated
- **Pie Charts:** Generally discouraged—bar charts are more informative

**Python Example:**
```python
df["category"].value_counts()           # Frequency table
df["category"].value_counts(normalize=True)  # Proportions
sns.countplot(data=df, x="category")    # Bar chart
```

---

### 7. Correlation and Relationships

Correlation measures the strength of the linear association between numerical variables.

#### Pearson Correlation Coefficient (\(r\))

\[
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{(n-1) s_x s_y}
\]

**Range:** −1 to +1

| Value | Interpretation |
|---|---|
| +1 | Perfect positive linear relationship |
| 0 | No linear relationship |
| −1 | Perfect negative linear relationship |

**Key Warnings:**
- **Correlation does not imply causation.** Two variables may move together without one causing the other.
- Pearson correlation measures **linear relationships only**; non-linear relationships may exist even when \(r \approx 0\).
- \(r\) is sensitive to outliers.

**Visualisation:**
- **Scatterplot:** Standard for two numeric variables
- **Hexagonal binning / Contour plots:** Better for large datasets
- **Correlation Heatmap:** Quick overview of all pairwise correlations

```python
df.corr(numeric_only=True)                    # Correlation matrix
sns.heatmap(df.corr(numeric_only=True), annot=True)  # Heatmap
sns.scatterplot(data=df, x="var1", y="var2")  # Scatterplot
```

---

### 8. Exploring Two or More Variables

| Relationship | Recommended Visualisation |
|---|---|
| Numeric vs Numeric | Scatterplot, hexbin plot, contour plot |
| Categorical vs Categorical | Contingency table, stacked bar chart |
| Numeric vs Categorical | Side-by-side boxplots, violin plots |
| Multiple Variables | Faceting / small multiples, colour/shape encoding |

**Python Examples:**
```python
# Numeric vs Categorical
sns.boxplot(data=df, x="category", y="value")
sns.violinplot(data=df, x="category", y="value")

# Multiple variables with faceting
sns.catplot(data=df, x="category", y="value", col="group")
```

---

## Important Formulas

### Location
\[
\text{Mean: } \bar{x} = \frac{\sum x_i}{n}
\]
\[
\text{Weighted Mean: } \bar{x}_w = \frac{\sum w_i x_i}{\sum w_i}
\]

### Variability
\[
\text{Variance: } s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}
\]
\[
\text{Standard Deviation: } s = \sqrt{s^2}
\]
\[
\text{IQR: } Q3 - Q1
\]
\[
\text{MAD: } \text{Median}(|x_i - \text{Median}(x)|)
\]

### Correlation
\[
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{(n-1) s_x s_y}
\]

---

## Machine Learning Relevance

EDA is critical in machine learning because it helps:

**Improve Feature Engineering:** Understanding distributions guides decisions about log transformations, normalisation, and scaling.

**Detect Bad Data:** EDA identifies missing values, duplicate records, outliers, and inconsistent categories before they corrupt models.

**Prevent Modelling Mistakes:** Without EDA, poor features are selected, wrong assumptions are made, and models overfit noisy data.

**Understand Feature Relationships:** Correlation analysis helps remove redundancy, reduce multicollinearity, and improve model stability.

---

## Common Pitfalls and Mistakes

1. **Using mean on skewed data** → Use median instead.
2. **Ignoring outliers** → Outliers may distort averages, break assumptions, and affect model performance. Investigate before removing.
3. **Assuming correlation means causation** → Always consider confounding variables.
4. **Skipping visualisation** → Always plot distributions before modelling; numbers alone can be misleading.
5. **Assuming normality** → Most real-world data is **not** normally distributed.
6. **Using wrong bin size in histograms** → Can obscure or create false patterns.
7. **Using pie charts** → Bar charts are almost always more effective for comparing proportions.
8. **Overlooking data types** → Treating categorical as numeric (or vice versa) leads to incorrect analyses.

---

## Chapter Summary

EDA is the foundation of data science. This chapter introduces methods to:

- Summarise data using measures of location and variability
- Understand distributions through visualisations
- Detect anomalies and outliers
- Explore relationships between variables
- Choose appropriate techniques based on data types

**Key Takeaway:** *Always understand the data before building models.*

---
## Connections to Other Chapters

- **Chapter 2 (Sampling Distributions):** Builds on concepts of variability
- **Chapter 3 (Hypothesis Testing):** Uses estimates derived from EDA
- **Chapter 4 (Regression):** Assumes understanding of correlation
- **Chapter 7 (PCA & Clustering):** Relies heavily on variability measures

---

## Progress Checklist

- [ ] Read complete chapter (pp. 1–46)
- [ ] Recreate all Python examples
- [ ] Complete `01_exploratory_data_analysis.ipynb`
- [ ] Solve all exercises in `exercises.ipynb`
- [ ] Perform experiments in `experiments.ipynb`

---