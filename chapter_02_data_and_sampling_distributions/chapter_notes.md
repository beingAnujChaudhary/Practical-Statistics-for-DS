# Chapter 02: Data and Sampling Distributions

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Data and sampling distributions form the foundation of statistical inference. Even in the era of big data, sampling principles remain essential for efficient analysis, bias reduction, and quantifying uncertainty in estimates.

In real-world data science, we rarely have access to an entire population. Instead, we work with samples and use them to draw conclusions about the broader population. This chapter bridges classical statistical theory (population-based inference) with modern resampling methods (the bootstrap) that are more aligned with data science practice.

**This chapter explains:**

- How data is distributed
- How sampling works and why randomness matters
- How uncertainty is measured
- How statisticians estimate population characteristics from samples
- Why these concepts are critical for machine learning models that must generalise to unseen data

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand the difference between population and sample
- Explain data distributions and sampling distributions
- Understand random sampling and recognise sources of sample bias
- Distinguish between bias and random error
- Explain the Central Limit Theorem (CLT) and its implications
- Calculate and interpret standard error
- Understand and construct confidence intervals
- Apply bootstrap resampling to estimate sampling distributions
- Recognise when normal, t-, binomial, chi-square, F-, and Poisson distributions are appropriate
- Apply distributional knowledge to real-world data science problems

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Population | Entire group of interest | Conceptual |
| Sample | Subset drawn from population | `df.sample()` |
| Random Sampling | Randomly selecting observations | `df.sample(frac=)` / `df.sample(n=)` |
| Sampling Bias | Systematic error in sampling | Conceptual (requires domain awareness) |
| Selection Bias | Bias from method of selecting observations | Conceptual |
| Sample Statistic | Value computed from sample | `mean()`, `median()`, etc. |
| Data Distribution | Distribution of individual values in a dataset | `sns.histplot()` |
| Sampling Distribution | Distribution of a sample statistic across repeated samples | Simulation |
| Central Limit Theorem | Sample means approach normality as n increases | Simulation |
| Standard Error | Variability of a sample statistic | `scipy.stats.sem()` |
| Bootstrap | Resampling technique for estimating sampling distributions | `sklearn.utils.resample()` |
| Confidence Interval | Range of plausible values for a population parameter | `scipy.stats.t.interval()` or bootstrap percentiles |
| Normal Distribution | Bell-shaped symmetric distribution | `scipy.stats.norm()` |
| t-Distribution | Normal-like with heavier tails; used for small samples | `scipy.stats.t()` |
| Binomial Distribution | Distribution of binary outcomes | `scipy.stats.binom()` |
| Poisson Distribution | Count data; events per unit time/space | `scipy.stats.poisson()` |

---

## Key Statistical Ideas

### 1. Population vs. Sample

**Population:** The entire set of items/observations of interest (often theoretical or inaccessible).
- *Examples:* All airline passengers, all customers, all transactions, all flights.
- Usually too expensive or impossible to measure completely.

**Sample:** A subset of the population used for analysis.
- *Example:* Instead of analysing 100 million customers, we analyse 10,000 randomly selected customers.
- **Goal:** The sample should represent the population.

```python
sample = df.sample(n=1000, random_state=42)
```

---

### 2. Random Sampling and Sample Bias

**Random Sampling:** Every observation has an equal chance of being selected.

- **With replacement:** Observations returned to population after selection.
- **Without replacement:** Observations removed after selection (more common in practice).

**Sample Bias:** Occurs when the sample systematically differs from the population in meaningful ways.

- *Classic example:* The Literary Digest 1936 poll sampled from telephone/automobile owners, overrepresenting affluent voters, and incorrectly predicted Landon over Roosevelt.

**Key Insight:** *Data quality (representativeness) often matters more than data quantity.* A biased large dataset is still misleading.

---

### 3. Selection Bias

**Definition:** Bias introduced by the method of selecting observations, consciously or unconsciously.

| Type | Description | Example |
|---|---|---|
| **Data Snooping** | Hunting through data until something "interesting" appears | Testing 20 predictors, reporting 1 "significant" by chance |
| **Vast Search Effect** | Repeated modelling/questions on large data increases false discoveries | Running hundreds of A/B tests, reporting only the "winner" |
| **Cherry-Picking** | Selecting time periods or subsets that support a conclusion | Showing only Q4 results when annual trend is negative |
| **Self-Selection** | Participants choose to respond, creating non-representative samples | Online reviews skewed toward extreme experiences |

**Guarding Against Selection Bias:**
- Pre-specify hypotheses and analysis plans
- Use holdout/validation sets to confirm findings
- Apply permutation tests (target shuffling) to assess validity
- Report all analyses conducted, not just "significant" results

---

### 4. Regression to the Mean

**Definition:** Extreme observations tend to be followed by more central ones in successive measurements.

**Mechanism:** When performance = skill + luck, selecting extreme performers captures both high skill AND good luck. In subsequent measurements, luck reverts toward the average.

**Examples:**
- "Rookie of the Year, sophomore slump" in sports
- Children of extremely tall parents tend to be shorter than their parents
- Stocks with extreme returns often revert toward the market average

**Implication for Data Science:** Be cautious interpreting extreme results—they may reflect random variation rather than a true effect.

---

### 5. Sampling Distribution of a Statistic

**Key Distinction:**

| Concept | Definition | Example |
|---|---|---|
| **Data Distribution** | Distribution of individual values in a dataset | Histogram of loan amounts |
| **Sampling Distribution** | Distribution of a statistic (e.g., mean) over many samples | Distribution of sample means from repeated sampling |

**Example:** Imagine repeatedly taking samples and calculating the mean income. Each sample gives a slightly different mean. These means create a *sampling distribution*. This concept is fundamental to confidence intervals, hypothesis testing, and statistical inference.

---

### 6. Central Limit Theorem (CLT)

One of the most important concepts in statistics.

**The CLT states:** *The distribution of sample means approaches normality as sample size increases, regardless of the population's shape.*

**Conditions:** Typically \(n \geq 30\) is considered sufficient (though this depends on the degree of skewness in the population).

**Why It Matters:** The CLT enables:
- Confidence intervals
- Significance testing
- Predictive modelling assumptions

Even if the original data is skewed, sample means become approximately normal with sufficient sample size.

---

### 7. Standard Error (SE)

Standard error measures the **uncertainty in a sample estimate**.

**Key Distinction:**
- **Standard Deviation (SD):** Measures spread of individual observations.
- **Standard Error (SE):** Measures spread of sample statistics across repeated samples.

\[
SE = \frac{s}{\sqrt{n}}
\]

**Interpretation:**
- Smaller standard error → more reliable estimate.
- To halve the SE, you must quadruple the sample size.

```python
from scipy.stats import sem
sem(df["column"])
```

---

### 8. The Bootstrap

**Core Idea:** Estimate the sampling distribution by resampling *with replacement* from the observed data, rather than collecting new samples.

**Algorithm for Bootstrap Resampling of the Mean:**
1. Draw a sample value from data, record it, replace it.
2. Repeat \(n\) times (same size as original sample).
3. Record the mean of this resample.
4. Repeat steps 1–3 \(R\) times (e.g., \(R = 1000\)).
5. Use the \(R\) bootstrap means to:
   - Estimate standard error (SD of bootstrap means)
   - Construct confidence intervals (percentiles of bootstrap distribution)
   - Visualise sampling distribution (histogram)

**Advantages:**
- No distributional assumptions required
- Works for virtually any statistic (mean, median, correlation, model parameters)
- Computationally feasible with modern computing

**Limitations:**
- Does not create new information or compensate for small sample size
- Assumes observed sample is representative of population
- May be unstable for very small samples (\(n < 20\))

```python
from sklearn.utils import resample
boot_sample = resample(df)
```

---

### 9. Confidence Intervals

**Definition:** An interval estimate that, under repeated sampling, would contain the true parameter value a specified percentage of the time (e.g., 95%).

**Bootstrap Confidence Interval Algorithm:**
1. Generate \(R\) bootstrap resamples.
2. Calculate the statistic of interest for each resample.
3. For an \(x\%\) confidence interval, trim \((100-x)/2\%\) from each end of the bootstrap distribution.
4. The remaining endpoints form the confidence interval.

**Correct Interpretation:** *"If we repeated this sampling procedure many times, 95% of such intervals would contain the true parameter."*

**Incorrect Interpretation:** *"There is a 95% probability the true parameter lies in this interval."* (After calculation, the interval either contains the true value or it doesn't—it's not a probability statement about the parameter.)

**Factors Affecting Interval Width:**
- Higher confidence level → wider interval
- Smaller sample size → wider interval
- Greater data variability → wider interval

```python
from scipy.stats import t
import numpy as np

confidence_interval = t.interval(
    confidence=0.95,
    df=len(data)-1,
    loc=np.mean(data),
    scale=stats.sem(data)
)
```

---

### 10. Normal Distribution

**Properties:**
- Bell-shaped, symmetric around mean \(\mu\)
- ~68% of data within \(\pm 1\sigma\), ~95% within \(\pm 2\sigma\)
- Defined by two parameters: mean (\(\mu\)) and standard deviation (\(\sigma\))

**Standard Normal Distribution:** \(\mu = 0\), \(\sigma = 1\). Values expressed as z-scores: \(z = \frac{x - \mu}{\sigma}\).

**QQ-Plots (Quantile-Quantile Plots):** Visual tool to assess if a sample follows a specified distribution. Points near the diagonal line indicate a good fit.

**Reality Check:** Most raw data is NOT normally distributed. Normality is more relevant for:
- Sampling distributions of statistics (via CLT)
- Model residuals (for inference validity)
- Error terms in certain models

---

### 11. Student's t-Distribution

**Characteristics:**
- Similar shape to normal but with **thicker tails**
- Family of distributions indexed by degrees of freedom (\(df\))
- As \(df \rightarrow \infty\), t-distribution approaches normal

**When to Use:**
- Estimating confidence intervals for means with small samples
- When population standard deviation is unknown (estimated from sample)
- Hypothesis tests comparing means (t-tests)

**Degrees of Freedom:** For a sample of size \(n\), \(df = n - 1\) (one constraint: sample mean).

---

### 12. Other Important Distributions

| Distribution | Use Case | Key Parameters |
|---|---|---|
| **Binomial** | Binary outcomes (click/no click, convert/not convert) | \(n\) = trials, \(p\) = success probability |
| **Poisson** | Count data; events per unit time/space (e.g., website visits/hour) | \(\lambda\) = rate |
| **Exponential** | Time between events in a Poisson process | \(\lambda\) = rate |
| **Chi-Square (\(\chi^2\))** | Testing independence in contingency tables; goodness-of-fit | Degrees of freedom |
| **F-Distribution** | ANOVA; comparing variances; regression model significance | Numerator df, denominator df |
| **Weibull** | Time-to-failure with changing hazard rate | Shape (\(\beta\)), scale (\(\eta\)) |

---

## Important Formulas

### Sampling and Estimation

\[
\text{Standard Error: } SE = \frac{s}{\sqrt{n}}
\]

\[
\text{Confidence Interval (normal approximation): } \bar{x} \pm z \left( \frac{s}{\sqrt{n}} \right)
\]

\[
\text{Z-score: } z = \frac{x - \mu}{\sigma}
\]

### Distribution Parameters

\[
\text{Binomial: Mean} = np, \quad \text{Variance} = np(1-p)
\]

\[
\text{Binomial PMF: } P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
\]

\[
\text{Poisson: Mean} = \lambda, \quad \text{Variance} = \lambda
\]

\[
\text{t-statistic: } t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
\]

---

## Python Implementation Notes

### Essential Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import resample  # for bootstrap resampling
```

### Key Functions

**Bootstrap Confidence Interval (from scratch):**

```python
def bootstrap_ci(data, statistic=np.mean, confidence=0.95, R=1000):
    """
    Compute bootstrap confidence interval for a given statistic.
    
    Parameters:
        data: array-like, the sample data
        statistic: callable, the statistic to compute
        confidence: float, confidence level (default 0.95)
        R: int, number of bootstrap resamples
    
    Returns:
        lower, upper, standard_error
    """
    bootstrap_stats = [
        statistic(resample(data, replace=True, n_samples=len(data)))
        for _ in range(R)
    ]
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    se = np.std(bootstrap_stats, ddof=1)
    return lower, upper, se

# Example usage
ci_lower, ci_upper, se = bootstrap_ci(df['value'].values)
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}], SE: {se:.3f}")
```

**QQ-Plot for Normality Assessment:**

```python
from scipy import stats
import matplotlib.pyplot as plt

def qq_plot(data, distribution=stats.norm):
    stats.probplot(data, dist=distribution, plot=plt)
    plt.title('QQ-Plot')
    plt.show()

# Check if sample means are approximately normal
sample_means = [df['value'].sample(n=30).mean() for _ in range(1000)]
qq_plot(sample_means)
```

**Distribution Functions:**

```python
# Binomial probabilities
stats.binom.pmf(k=5, n=20, p=0.3)   # P(X = 5)
stats.binom.cdf(k=5, n=20, p=0.3)   # P(X ≤ 5)

# Normal distribution
stats.norm.cdf(x=1.96)               # P(Z ≤ 1.96) ≈ 0.975
stats.norm.ppf(0.975)                # z-value for 97.5th percentile ≈ 1.96

# Chi-square test for independence
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

**Visualising Sampling Distributions vs. Data Distribution:**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original data distribution
axes[0].hist(df['value'], bins=30, edgecolor='black')
axes[0].set_title('Original Data Distribution')

# Sampling distribution (n=5)
means_n5 = [df['value'].sample(n=5).mean() for _ in range(1000)]
axes[1].hist(means_n5, bins=30, edgecolor='black')
axes[1].set_title('Sampling Distribution (n=5)')

# Sampling distribution (n=30)
means_n30 = [df['value'].sample(n=30).mean() for _ in range(1000)]
axes[2].hist(means_n30, bins=30, edgecolor='black')
axes[2].set_title('Sampling Distribution (n=30)')

plt.tight_layout()
plt.show()
```

---

## Machine Learning Relevance

### Train/Test Splits Are Sampling

Machine learning models train on sampled subsets, not full populations. Poor sampling causes bias, poor generalisation, and overfitting.

### Bootstrap in Ensemble Learning

Methods like bagging and random forests use bootstrap sampling internally to create diverse training sets and reduce overfitting.

### Understanding Uncertainty

Confidence intervals help evaluate model reliability, performance variation, and uncertainty in estimates—crucial for model deployment decisions.

### Distribution Awareness

Many ML models assume approximately normal features. Feature distributions often require scaling, transformations, or log transformations. Understanding the underlying distribution guides preprocessing choices.

---

## Common Pitfalls and Mistakes

1. **Confusing population with sample:** Sample statistics are estimates, not exact truth.
2. **Ignoring sampling bias:** Random sampling matters more than sample size. Large but biased samples yield precise but inaccurate estimates.
3. **Misinterpreting confidence intervals:** They describe long-run sampling behaviour, not single-event probability.
4. **Assuming all data is normally distributed:** Real-world data is often skewed, heavy-tailed, and messy.
5. **Confusing standard deviation with standard error:** SD describes spread of observations; SE describes spread of estimates.
6. **Ignoring regression to the mean:** Attributing natural reversion to interventions or treatments.
7. **Using t-distribution inappropriately:** Requires approximately normal sampling distribution; bootstrap may be safer for small, skewed samples.
8. **Forgetting bootstrap limitations:** Cannot compensate for fundamental sampling bias or extremely small samples.

---

## Key Takeaways

1. **Sampling matters even with big data:** Random sampling reduces bias and enables efficient analysis; quality beats quantity.
2. **Distinguish data vs. sampling distributions:** Individual values follow the data distribution; statistics follow sampling distributions.
3. **Central Limit Theorem is powerful but not universal:** Enables normal approximations for many statistics, but requires adequate sample size.
4. **Bootstrap is the data scientist's friend:** Flexible, assumption-light method for estimating uncertainty in virtually any statistic.
5. **Confidence intervals quantify uncertainty:** Wider intervals reflect less precision; interpret correctly (frequentist interpretation).
6. **Distribution choice depends on data type and question:** Binary outcomes → Binomial; Counts → Poisson; Times between events → Exponential.
7. **Regression to the mean is ubiquitous:** Extreme results often revert; avoid over-interpreting single extreme observations.
8. **Selection bias is insidious:** Pre-specify analyses, use validation sets, and report all tests to avoid false discoveries.

---

## Connections to Other Chapters

- **Chapter 1:** EDA tools (histograms, boxplots) visualise distributions discussed here.
- **Chapter 3:** Hypothesis testing builds on sampling distributions, p-values, and confidence intervals.
- **Chapter 4:** Regression inference relies on t-distributions and standard errors of coefficients.
- **Chapter 5:** Binomial distribution underlies classification probability estimates.
- **Chapter 6:** Bootstrap aggregating (bagging) extends bootstrap ideas to ensemble modelling.

---

### Questions I Still Have
- How does the bootstrap perform with my actual dataset vs. simulated data?
- When should I prefer bootstrap CI over formula-based CI in practice?
- What sample size is "large enough" for CLT to apply to my skewed business metrics?
- How can I detect selection bias in observational data where random sampling wasn't possible?


## Progress Checklist

- [ ] Read complete chapter (pp. 47–86)
- [ ] Recreate all Python examples
- [ ] Implement bootstrap CI function from scratch
- [ ] Complete `02_data_and_sampling_distributions.ipynb`
- [ ] Solve all exercises in `exercises.ipynb`
- [ ] Experiment with different sample sizes/distributions in `experiments.ipynb`

---
