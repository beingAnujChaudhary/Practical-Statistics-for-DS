# Chapter 03: Statistical Experiments and Significance Testing

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Statistical experiments and significance testing help us determine whether observed patterns in data are meaningful or simply due to random chance. This chapter bridges classical statistical inference with modern data science practice, emphasising **resampling and permutation methods** as flexible, assumption-light alternatives to traditional formula-based hypothesis tests.

In data science and machine learning, we often compare:

- Two website designs
- Marketing campaigns
- Recommendation systems
- Model performance
- Product changes

**The central question is always:** *Is the observed difference real or random?*

The core message for data scientists: understand uncertainty, guard against being fooled by random chance, and use experiments (especially A/B tests and multi-arm bandits) to make data-driven decisions.

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Design and interpret A/B tests with proper control groups and randomisation
- Formulate null and alternative hypotheses for business and experimental questions
- Implement permutation tests to assess statistical significance without distributional assumptions
- Correctly interpret p-values and recognise their limitations
- Distinguish between statistical significance and practical significance
- Identify and mitigate multiple testing (alpha inflation) and false discovery risks
- Apply ANOVA and chi-square tests for group comparisons and categorical independence
- Understand when and how to use multi-arm bandit algorithms for sequential testing
- Calculate or simulate power and required sample size for experiments

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Experiment | Controlled comparison between groups | Conceptual |
| A/B Test | Compare two variants (control vs. treatment) | `df.groupby()` |
| Null Hypothesis (H₀) | No effect assumption | Conceptual |
| Alternative Hypothesis (H₁) | Effect exists | Conceptual |
| Statistical Significance | Difference unlikely by chance alone | p-value |
| p-Value | Probability of observing results this extreme, assuming H₀ is true | `scipy.stats` / permutation test |
| Permutation Test | Resampling-based significance test | Custom function / `mlxtend.evaluate` |
| t-Test | Compare means of two groups | `scipy.stats.ttest_ind()` |
| ANOVA | Compare means of 3+ groups | `scipy.stats.f_oneway()` |
| Chi-Square Test | Test independence of categorical variables | `scipy.stats.chi2_contingency()` |
| Type I Error (False Positive) | Rejecting H₀ when it is true | Controlled by α |
| Type II Error (False Negative) | Failing to reject H₀ when effect exists | Related to power |
| Statistical Power | Probability of detecting a real effect | Simulation / `statsmodels` |
| Multiple Testing Problem | Inflated false positives from many tests | Bonferroni, FDR correction |
| Multi-Arm Bandit | Dynamic traffic allocation across variants | Epsilon-greedy, Thompson sampling |

---

## Key Statistical Ideas

### 1. Statistical Experiments and A/B Testing

A statistical experiment compares outcomes between groups to determine whether differences are caused by real effects or random variation.

**A/B Testing Structure:**
- **Group A (Control):** Current version, baseline condition.
- **Group B (Treatment):** New version, experimental condition.
- **Randomisation:** Subjects randomly assigned to ensure comparability and isolate the treatment effect.

**Common Metrics Compared:**
- Conversion rates
- Click-through rates
- Revenue per user
- Engagement time

```python
group_stats = df.groupby("group")["conversion"].mean()
```

**Why Control Groups Matter:** They isolate the treatment effect from external factors and prevent confounding.

---

### 2. Hypothesis Testing Framework

Hypothesis testing starts with clearly stated assumptions.

**Null Hypothesis (H₀):** Default assumption — *no difference exists.*
- *Example:* "The new website has no impact on conversions."

**Alternative Hypothesis (H₁):** Competing claim — *a difference exists.*
- *Example:* "The new website improves conversion rate."

**Decision Process:** Evidence leads us to either *reject H₀* or *fail to reject H₀*. We never truly "prove" hypotheses.

**One-Way vs. Two-Way Tests:**
- **One-way:** Tests for improvement in one direction only.
- **Two-way:** Tests for any difference (more common in practice).

---

### 3. Resampling and Permutation Tests

Permutation testing is a modern, intuitive, and assumption-light significance test.

**Core Idea:** Shuffle group labels randomly, recalculate the test statistic, and compare the observed difference to the distribution of permuted differences.

**Algorithm:**
1. Combine outcomes from all groups.
2. Randomly shuffle group labels.
3. Split into groups matching original sizes.
4. Calculate the difference in the statistic (e.g., mean difference).
5. Repeat \(R\) times (e.g., 1,000–10,000).
6. Compare the observed statistic to the permuted distribution.

**Advantages:**
- No normality assumptions required
- Works with any test statistic
- Handles unbalanced samples
- Intuitive interpretation

```python
def permutation_test(group_a, group_b, statistic_func=np.mean, R=10000):
    observed = statistic_func(group_a) - statistic_func(group_b)
    combined = np.concatenate([group_a, group_b])
    perm_diffs = []
    for _ in range(R):
        np.random.shuffle(combined)
        perm_a = combined[:len(group_a)]
        perm_b = combined[len(group_a):]
        perm_diffs.append(statistic_func(perm_a) - statistic_func(perm_b))
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed))
    return observed, p_value, perm_diffs
```

---

### 4. Statistical Significance and p-Values

**Statistical Significance:** Measures whether results are unlikely to occur by random chance alone.

**Common Threshold:** \(\alpha = 0.05\)

If \(p < 0.05\), results are often called *statistically significant*.

**p-Value Definition:** *The probability of observing results as extreme as (or more extreme than) the actual results, assuming the null hypothesis is true.*

**Interpretation:**
- **Small p-value:** Evidence against H₀.
- **Large p-value:** Weak evidence against H₀.

**Critical Misconceptions (ASA Warning, 2016):**
- A p-value is **NOT** the probability that H₀ is true.
- A p-value is **NOT** the probability that results happened by chance.
- Statistical significance does **NOT** imply practical importance.
- A p-value alone should **not** be the sole basis for decisions.

---

### 5. Type I and Type II Errors

| Error Type | Definition | Example | Controlled By |
|---|---|---|---|
| **Type I (False Positive)** | Rejecting H₀ when it is actually true | Concluding new website works when it doesn't | α (alpha), typically 0.05 |
| **Type II (False Negative)** | Failing to reject H₀ when an effect truly exists | Missing a genuinely better product design | Power, sample size |

**Statistical Power:** The probability of detecting a real effect when it exists.

\[
\text{Power} = 1 - \beta \quad \text{(where } \beta \text{ is the Type II error rate)}
\]

Power increases with:
- Larger sample size
- Stronger effect size
- Lower variability

---

### 6. t-Tests

t-tests compare means between two groups when data is numeric.

**t-Statistic:** Standardised difference between means, scaled by pooled standard error.

\[
t = \frac{\bar{x}_1 - \bar{x}_2}{SE_{\text{pooled}}}
\]

**Types:**
- **Independent t-test:** Separate groups (e.g., website A vs. website B).
- **Paired t-test:** Same subjects measured twice (e.g., before vs. after intervention).

**Data Science Relevance:** Useful for quick comparisons, but resampling is often preferred for flexibility.

```python
from scipy.stats import ttest_ind
ttest_ind(group_a, group_b, equal_var=False)
```

---

### 7. ANOVA (Analysis of Variance)

**Purpose:** Test whether means of three or more groups differ significantly.

**F-Statistic:** Ratio of between-group variance to within-group variance.

**Decomposition of Variance:** Total variation = treatment effect + residual error + interaction effects.

**Resampling Alternative:** Shuffle group labels, compute the variance ratio, repeat — estimate p-value without normality assumptions.

```python
from scipy.stats import f_oneway
f_oneway(group_a, group_b, group_c)
```

---

### 8. Chi-Square Test

**Purpose:** Test independence between categorical variables or goodness-of-fit to an expected distribution.

**Statistic:**
\[
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
\]

**Resampling Approach:** Shuffle category labels, recompute chi-square, compare to observed.

**Fisher's Exact Test:** Preferred for small counts (expected < 5); computes exact probability under the null.

**Data Science Applications:** Feature selection, spatial analysis, fraud detection (digit distribution anomalies).

```python
from scipy.stats import chi2_contingency
observed = np.array([[a_yes, a_no], [b_yes, b_no]])
chi2, p, dof, expected = chi2_contingency(observed)
```

---

### 9. Multiple Testing Problem

**Problem:** Testing many hypotheses increases the probability of false positives.

**Example:** 20 independent tests at \(\alpha = 0.05\) → ~64% chance of at least one false positive.

**Mitigation Strategies:**
- Holdout/validation sets for model selection
- **Bonferroni correction:** \(\alpha_{\text{adjusted}} = \alpha / m\) (where \(m\) = number of tests)
- **Benjamini-Hochberg FDR:** Controls the false discovery rate
- Pre-specify hypotheses; avoid "data dredging"
- Use cross-validation for predictive modelling

---

### 10. Multi-Arm Bandit Algorithms

**Problem with Fixed A/B Tests:** Wastes traffic on inferior treatments; slow to converge.

**Bandit Solution:** Dynamically allocate traffic to better-performing arms while still exploring alternatives.

**Epsilon-Greedy:** With probability \(\varepsilon\), explore randomly; otherwise, exploit the best arm.

**Thompson Sampling:** Bayesian approach; sample from posterior distributions of each arm, pick the arm with the highest sampled value.

**Advantages:** Faster optimisation, handles 3+ arms efficiently, minimises regret.

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.values))
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
```

---

### 11. Power and Sample Size

**Power:** Probability of detecting a true effect of a specified size.

**Components:** Effect size, sample size, \(\alpha\) level, variance.

**Simulation Approach for Data Science:**
1. Create synthetic data reflecting expected distributions.
2. Add target effect size.
3. Bootstrap resample, run test, record significance.
4. Repeat → estimate power.

**Rule of Thumb:** Rare events require larger samples; power calculations prevent inconclusive experiments. To halve standard error, quadruple sample size.

```python
def simulate_power(effect_size, n_per_group, alpha=0.05, R=1000):
    significant = 0
    for _ in range(R):
        g1 = np.random.normal(0, 1, n_per_group)
        g2 = np.random.normal(effect_size, 1, n_per_group)
        _, p = stats.ttest_ind(g1, g2)
        if p < alpha:
            significant += 1
    return significant / R
```

---

## Important Formulas

### Hypothesis Testing

\[
\text{p-value} = P(\text{statistic} \geq \text{observed} \mid H_0 \text{ true}) \approx \frac{\# \text{permuted stats} \geq \text{observed}}{R}
\]

\[
t = \frac{\bar{x}_1 - \bar{x}_2}{SE_{\text{pooled}}}, \quad SE_{\text{pooled}} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
\]

### ANOVA and Chi-Square

\[
F = \frac{MS_{\text{between}}}{MS_{\text{within}}}
\]

\[
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}, \quad df = (\text{rows} - 1)(\text{cols} - 1)
\]

### Power and Sample Size

\[
\text{Power} = 1 - \beta, \quad SE = \frac{s}{\sqrt{n}} \rightarrow \text{Quadruple } n \text{ to halve SE}
\]

---

## Common Visualisations

### A/B Comparison Plots

Used for comparing conversion rates and group performance.

```python
sns.barplot(data=df, x="group", y="conversion")
```

### Boxplots for Group Comparison

Used for comparing group variability and identifying outliers.

```python
sns.boxplot(data=df, x="group", y="metric")
```

### Permutation Distribution Plots

Used to visualise the null distribution after shuffling. Compare observed statistic to the histogram of permuted statistics.

---

## Machine Learning Relevance

### A/B Testing for Models

Used to compare ranking algorithms, recommendation systems, search quality, and pricing models in production.

### Model Comparison

Significance testing helps determine whether Model A genuinely outperforms Model B or if differences are due to random variation.

### Feature Selection

Hypothesis testing helps identify useful predictors. Chi-square tests are commonly used for categorical feature selection.

### Product Experimentation

Modern technology companies (Google, Netflix, Amazon, Meta) rely heavily on online experimentation for data-driven decisions.

---

## Common Pitfalls and Mistakes

1. **Misinterpreting p-values:** p-value ≠ probability H₀ is true; p-value ≠ probability results are due to chance.
2. **Confusing statistical significance with practical importance:** A tiny effect can be statistically significant with enough data but may have no business value.
3. **P-hacking / Data dredging:** Repeatedly testing until \(p < 0.05\) creates misleading conclusions.
4. **Stopping experiments early:** When results "look significant," this inflates Type I error.
5. **Ignoring multiple testing:** Running many A/B tests or feature significance checks without correction guarantees false positives.
6. **Using chi-square with small expected counts** (< 5): Switch to Fisher's exact test or resampling.
7. **Confusing correlation with causation:** In observational A/B-like analyses without randomisation.
8. **Over-optimising bandits** with too-low \(\varepsilon\): Premature convergence to suboptimal arms.

---

## Key Takeaways

1. **Resampling beats rigid formulas:** Permutation and bootstrap tests are flexible, assumption-light, and align better with modern data science workflows.
2. **p-Values are not proof:** A low p-value only indicates incompatibility with H₀; it does not measure effect size, practical importance, or the probability that H₁ is true.
3. **Multiple testing is dangerous:** Always adjust for multiple comparisons or use holdout sets; data dredging guarantees false positives.
4. **A/B testing requires rigour:** Randomisation, pre-specified metrics, and control groups are non-negotiable for valid causal inference.
5. **Bandits optimise continuously:** Multi-arm bandits outperform fixed A/B tests in dynamic environments by balancing exploration and exploitation.
6. **Power planning prevents waste:** Simulate or calculate required sample size before launching experiments; underpowered tests yield inconclusive results.
7. **Statistical ≠ practical significance:** Large samples detect trivial effects; always tie results to business impact.

---

## Connections to Other Chapters

- **Chapter 1:** EDA informs metric selection for A/B tests; boxplots and histograms visualise group differences.
- **Chapter 2:** Sampling distributions, bootstrap, and standard error underpin permutation test logic.
- **Chapter 4:** t-tests appear in regression coefficient significance; ANOVA generalises to linear models.
- **Chapter 5:** Chi-square tests used for feature selection; A/B testing informs classification threshold tuning.
- **Chapter 6:** Bandit algorithms extend to model selection and hyperparameter tuning; cross-validation mitigates multiple testing.

---

### Questions I Still Have
- How does permutation test performance scale with dataset size compared to classical approximations?
- When should I prefer Thompson sampling over epsilon-greedy in production bandits?
- How to properly adjust for multiple testing in automated feature selection pipelines?
- What's the minimum detectable effect size for my current business metric given traffic volume?



## Progress Checklist

- [ ] Read complete chapter (pp. 87–139)
- [ ] Implement permutation test from scratch
- [ ] Compare resampling p-values with `scipy.stats` classical tests
- [ ] Complete `03_statistical_experiments_and_significance_testing.ipynb`
- [ ] Solve all exercises in `exercises.ipynb`
- [ ] Build epsilon-greedy bandit simulation in `experiments.ipynb`
- [ ] Simulate power/sample size for a binary conversion metric

---
