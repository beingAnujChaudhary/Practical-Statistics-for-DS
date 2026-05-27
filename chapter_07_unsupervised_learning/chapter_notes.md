# Chapter 07: Unsupervised Learning

> **Source:** *Practical Statistics for Data Scientists, 2nd Edition* by Peter Bruce, Andrew Bruce, and Peter Gedeck

---

## Chapter Overview

Unsupervised learning refers to statistical methods that extract meaning from data without training a model on labelled data. Unlike supervised learning (regression, classification), unsupervised learning does not distinguish between response and predictor variables. Instead, it discovers hidden patterns, reduces dimensionality, or identifies meaningful groups within the data.

**The goal is to:**
- Discover hidden patterns
- Identify natural groupings
- Reduce dimensionality
- Detect anomalies
- Summarise complex data

**Unsupervised learning is especially useful when no labelled target variable exists.**

**Examples:**
- Customer segmentation
- Fraud detection
- Recommendation systems
- Market basket analysis
- Anomaly detection

**This chapter covers three core unsupervised techniques:**
1. **Principal Components Analysis (PCA):** Reduces dimensionality by finding linear combinations of variables that explain maximum variance.
2. **K-Means Clustering:** Partitions data into K groups by minimising within-cluster variance.
3. **Hierarchical Clustering:** Builds a tree of clusters without pre-specifying K.
4. **Model-Based Clustering:** Uses probabilistic mixture models for soft cluster assignments.

Additionally, we address practical considerations: scaling variables, handling categorical data, and selecting the number of clusters.

---

## Learning Objectives

By the end of this chapter, I should be able to:

- Understand unsupervised learning fundamentals and distinguish it from supervised learning
- Apply PCA to reduce dimensionality and interpret principal component loadings
- Use scree plots and cumulative variance to select significant components
- Implement K-means clustering and interpret cluster assignments and centroids
- Apply the elbow method and other criteria to select the number of clusters
- Perform hierarchical clustering and interpret dendrograms
- Compare complete, single, average linkage, and Ward's methods
- Apply model-based clustering using Gaussian mixture models
- Scale and normalise variables appropriately for unsupervised methods
- Handle categorical variables using Gower's distance or appropriate encoding
- Use unsupervised learning as a building block for supervised prediction

---

## Key Concepts

| Concept | Description | Python Implementation |
|---|---|---|
| Unsupervised Learning | Learning without labelled outcomes | `scikit-learn` |
| Clustering | Grouping similar observations together | `KMeans()`, `AgglomerativeClustering()` |
| K-Means | Partition-based clustering minimising within-cluster variance | `KMeans()` |
| Hierarchical Clustering | Tree-based clustering; agglomerative or divisive | `AgglomerativeClustering()`, `scipy.cluster.hierarchy` |
| Dendrogram | Visualisation of cluster hierarchy | `dendrogram()` |
| PCA | Dimensionality reduction via linear combinations | `PCA()` |
| Principal Component | Direction of maximum variance in the data | `pca.components_` |
| Loadings | Weights showing original variable contributions to components | `pca.components_.T` |
| Explained Variance | Proportion of total variance captured by each component | `explained_variance_ratio_` |
| Model-Based Clustering | Probabilistic clustering via mixture models | `GaussianMixture()` |
| Gower's Distance | Distance metric for mixed numeric and categorical data | `prince.gower.gower_matrix()` |
| Similarity/Distance | Measure of closeness between observations | Euclidean, Manhattan, Mahalanobis |

---

## Key Statistical Ideas

### 1. What Is Unsupervised Learning?

Unsupervised learning attempts to find hidden structure in data without labelled outcomes.

| | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Inputs** | Features + Labels | Features only |
| **Goal** | Predict known outcomes | Discover hidden patterns |
| **Example** | Customer data → churn prediction | Customer data → hidden customer groups |
| **Evaluation** | Accuracy, RMSE, AUC | Silhouette score, domain validation |

**Key questions unsupervised learning answers:** *What groups naturally exist in this data? What underlying structure can we discover?*

---

### 2. Principal Components Analysis (PCA)

PCA is a dimensionality reduction technique that transforms many correlated numeric variables into a smaller set of uncorrelated principal components that capture most of the total variance.

**Mathematical Form — First Principal Component:**
\[
Z_1 = w_{1,1}X_1 + w_{1,2}X_2 + \cdots + w_{1,p}X_p
\]
Where weights \(w\) are chosen to maximise variance explained, subject to:
- Components are orthogonal (uncorrelated with each other)
- Weights are normalised: \(\sum w_{1,j}^2 = 1\)

**Key Properties:**
- PC1 explains the most variance; PC2 explains the most remaining variance orthogonal to PC1, and so on.
- Loadings (weights) reveal which original variables contribute most to each component.
- The sign of loadings is arbitrary; focus on relative magnitudes and patterns.
- Components may represent latent constructs (e.g., "market trend," "risk factor").

**Interpretation Guidelines:**
- Examine loadings: large absolute values indicate important variable contributions.
- Use scree plots (variance explained per component) and cumulative variance plots to decide how many components to retain.
- PCA works only with numeric variables; categorical data must be converted first.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Always standardise before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

pca = PCA(n_components=None)
pca.fit(X_scaled)

# Explained variance
print(pca.explained_variance_ratio_)

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_numeric.columns
)

# Transform data
X_pca = pca.transform(X_scaled)
```

**Benefits:**
- Simplifies data for visualisation
- Reduces noise and computational burden
- Creates uncorrelated features for downstream modelling

---

### 3. K-Means Clustering

K-means is one of the most popular clustering algorithms. It partitions \(n\) records into \(K\) clusters by minimising the within-cluster sum of squares (WCSS):

\[
\text{Minimise: } \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2
\]
Where \(\mu_k\) is the centroid (mean vector) of cluster \(C_k\).

**Algorithm:**
1. Initialise \(K\) cluster centroids (randomly or via k-means++).
2. Assign each record to the nearest centroid (Euclidean distance).
3. Recompute centroids as the mean of assigned records.
4. Repeat steps 2–3 until assignments stabilise.

**Key Considerations:**
- Requires pre-specifying \(K\) (number of clusters).
- Sensitive to initial centroid placement; use multiple random starts (`n_init`).
- Assumes clusters are spherical and of similar size/density.
- **Standardisation is essential:** Variables with large scales dominate distance calculations.

```python
from sklearn.cluster import KMeans

# Always standardise first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Centroids in original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
```

**Advantages:** Simple, fast, scalable to large datasets.
**Limitations:** Must choose \(K\) beforehand, sensitive to scaling, struggles with irregular cluster shapes.

---

### 4. Choosing the Number of Clusters (K)

Determining the appropriate number of clusters is one of the most challenging aspects of clustering.

**Methods for Selecting K:**

| Method | Description | How to Use |
|---|---|---|
| **Elbow Method** | Plot WCSS vs. K; look for "elbow" where marginal gain diminishes | Visual inspection |
| **Silhouette Score** | Measures how similar records are to their own cluster vs. other clusters | Higher is better; range [−1, 1] |
| **Gap Statistic** | Compares observed WCSS to reference null distribution | Statistical test |
| **Cross-Validation** | Assess stability of clusters on holdout data | Practical for large datasets |
| **Domain Knowledge** | Practical constraints often dictate reasonable K | Essential for business application |

**Important:** No single metric is definitive. Use multiple criteria together and validate clusters with domain expertise.

```python
# Elbow method
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'o-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method')
plt.show()

# Silhouette score
from sklearn.metrics import silhouette_score
silhouette = silhouette_score(X_scaled, clusters)
```

---

### 5. Hierarchical Clustering

Hierarchical clustering creates a nested cluster structure by iteratively merging (agglomerative) or splitting (divisive) records based on pairwise dissimilarity.

**Agglomerative Algorithm (Most Common):**
1. Start with each record as its own cluster.
2. Compute dissimilarity between all cluster pairs.
3. Merge the two most similar clusters.
4. Repeat until all records form one cluster.

**Linkage Methods (How to Measure Cluster Dissimilarity):**

| Method | Formula | Characteristics |
|---|---|---|
| **Single Linkage** | \(\min d(a_i, b_j)\) | "Chaining" effect; sensitive to noise |
| **Complete Linkage** | \(\max d(a_i, b_j)\) | Compact, well-separated clusters |
| **Average Linkage** | \(\text{mean } d(a_i, b_j)\) | Compromise between single and complete |
| **Ward's Method** | Minimise \(\Delta\)(WCSS) | Similar to K-means; minimises variance within clusters |

**Dendrogram Interpretation:**
- Leaves = individual records.
- Branch height = dissimilarity at merge.
- A horizontal cut at height \(h\) yields clusters where all within-cluster dissimilarities < \(h\).
- No need to pre-specify \(K\); explore multiple cuts visually.

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=X_numeric.index, leaf_rotation=90)
plt.axhline(y=15, color='red', linestyle='--', label='Cut at height=15')
plt.legend()
plt.show()

# Extract clusters
clusters = fcluster(Z, t=4, criterion='maxclust')
```

**Advantages:** Interpretable dendrograms, no strict K requirement initially.
**Limitations:** Computationally expensive (O(n³) time, O(n²) memory); not suitable for very large datasets (>10k records). Merges are irreversible.

---

### 6. Distance and Similarity

Clustering depends heavily on distance measures. Smaller distance means more similarity.

**Common Distance Metrics:**

| Metric | Formula | Use Case |
|---|---|---|
| **Euclidean** | \(\sqrt{\sum (x_i - y_i)^2}\) | Default for continuous variables |
| **Manhattan** | \(\sum |x_i - y_i|\) | Grid-like distances; robust to outliers |
| **Mahalanobis** | \(\sqrt{(x-y)^T \Sigma^{-1} (x-y)}\) | Accounts for covariance among variables |

**Critical Warning:** Feature scaling matters enormously. Without standardisation, variables like salary (e.g., 100,000) dominate distance calculations over variables like age (e.g., 30). Always standardise features before distance-based clustering.

**Gower's Distance for Mixed Data:**
- Handles numeric + categorical variables in one distance matrix.
- Numeric variables: scaled Manhattan distance (range 0–1).
- Categorical variables: 0 if same category, 1 if different.
- Overall distance = weighted average across all variables.

---

### 7. Model-Based Clustering

Instead of distance-based rules, model-based clustering uses probabilistic assumptions. Data is assumed to be generated from a mixture of \(K\) probability distributions (typically multivariate normal).

**Mixture Model:**
\[
f(x) = \sum_{k=1}^{K} \pi_k \cdot N_p(x \mid \mu_k, \Sigma_k)
\]
Where \(\pi_k\) are mixing proportions (\(\sum \pi_k = 1\)).

**Advantages:**
- Statistically principled; uses likelihood and AIC/BIC for model selection.
- Flexible covariance structures: spherical, diagonal, ellipsoidal.
- Provides **soft clustering** — probabilistic cluster assignments rather than hard assignments.
- Handles clusters of different shapes, sizes, and orientations.

**Model Selection:** Use BIC (Bayesian Information Criterion) to balance fit vs. complexity. Lower BIC = better model.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# Hard and soft assignments
hard_clusters = gmm.predict(X_scaled)
soft_clusters = gmm.predict_proba(X_scaled)

print(f"BIC: {gmm.bic(X_scaled):.1f}")
```

**Limitations:** Computationally intensive; assumes parametric form (normality); sensitive to initialisation.

---

### 8. Handling Categorical Variables in Clustering

| Approach | When to Use | Caveats |
|---|---|---|
| **One-Hot Encoding** | Few categories, K-means/PCA | Creates high dimensionality; binary variables may dominate |
| **Gower's Distance** | Mixed data, hierarchical clustering | Not compatible with all algorithms; computationally heavier |
| **Separate Clustering** | Many categories | Loses cross-type interactions |
| **Target Encoding** | Supervised downstream task | Risk of overfitting; requires outcome variable |

**For K-means with one-hot encoded categories:** Scale ALL variables (both numeric and dummy binaries) after encoding, but be aware that binary variables may still dominate.

---

## Important Formulas

### PCA
\[
\text{Principal Component: } Z_k = w_{k,1}X_1 + \cdots + w_{k,p}X_p, \quad \sum_j w_{k,j}^2 = 1
\]
\[
\text{Variance explained by PC}_k: \frac{\lambda_k}{\sum_j \lambda_j}
\]

### K-Means
\[
\text{Within-Cluster SS: } WCSS = \sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2
\]
\[
\text{Centroid Update: } \mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i
\]

### Hierarchical Clustering Linkage
\[
\text{Single: } D(A,B) = \min_{i \in A, j \in B} d(x_i, x_j)
\]
\[
\text{Complete: } D(A,B) = \max_{i \in A, j \in B} d(x_i, x_j)
\]
\[
\text{Average: } D(A,B) = \frac{1}{|A||B|} \sum_{i \in A} \sum_{j \in B} d(x_i, x_j)
\]
\[
\text{Ward: } \Delta(A,B) = WCSS(A \cup B) - [WCSS(A) + WCSS(B)]
\]

### Model-Based Clustering
\[
\text{Mixture Density: } f(x) = \sum_{k=1}^{K} \pi_k \cdot N(x \mid \mu_k, \Sigma_k)
\]
\[
\text{BIC} = -2 \cdot \log(L) + p \cdot \log(n) \quad \text{[lower is better]}
\]

### Gower's Distance
\[
\text{Numeric: } d_j = \frac{|x_i - x_j|}{R_j} \quad (R_j = \text{range}), \quad \text{Categorical: } d_j = 0 \text{ if same, } 1 \text{ if different}
\]
\[
\text{Overall: } D = \frac{1}{p} \sum_j d_j
\]

---

## Common Visualisations

### Scree Plot (PCA)
Shows proportion of variance explained by each principal component. Used to select number of components.
```python
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), 
         pca.explained_variance_ratio_, 'o-')
```

### Cluster Scatterplots
Used to inspect group separation after clustering or PCA projection.
```python
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set2')
```

### Dendrogram
Visualises hierarchical cluster structure; useful for choosing cluster count.
```python
dendrogram(linkage_matrix)
```

### Elbow Plot
Used to determine optimal K in K-means by plotting WCSS against K.

### Cluster Centroid Profiles
Plot cluster means for each variable to interpret and name clusters meaningfully.

---

## Machine Learning Relevance

Unsupervised learning is widely used in production systems:

- **Customer Segmentation:** Group customers based on behaviour, spending, and engagement patterns.
- **Fraud / Anomaly Detection:** Detect unusual behaviour without labelled examples of fraud.
- **Recommendation Systems:** Identify similar users and similar products.
- **Data Compression:** PCA reduces dimensions before modelling, speeding computation and reducing noise.
- **Feature Engineering:** Cluster labels and PCA components can become new predictive features for supervised models.

---

## Common Pitfalls and Mistakes

1. **Ignoring feature scaling:** Distance-based methods (K-means, hierarchical) and PCA require standardisation. Unscaled variables with large ranges will dominate results.
2. **Choosing K randomly:** Use elbow method, silhouette score, gap statistic, and domain knowledge together — no single metric is definitive.
3. **Overinterpreting clusters:** Clusters are discovered patterns, not guaranteed truth. Validate with domain expertise and stability analysis.
4. **Using PCA blindly:** Reduced dimensions may lose important interpretability; examine loadings to understand what components represent.
5. **Assuming clusters always exist:** Some datasets have no meaningful cluster structure; clustering will still produce groups.
6. **Using Euclidean distance with mixed data types:** Numeric and categorical variables require different distance metrics (Gower's distance).
7. **Applying K-means to non-spherical clusters:** K-means assumes spherical, equal-sized clusters; model-based methods may be more appropriate for irregular shapes.
8. **Ignoring cluster size imbalance:** Very small clusters may represent outliers or noise rather than meaningful segments.

---

## Key Takeaways

1. **Unsupervised learning discovers structure without labels:** Use PCA for dimensionality reduction, clustering for segmentation, and model-based methods for probabilistic assignments.
2. **Scaling is non-negotiable:** Always standardise numeric variables before distance-based methods.
3. **No single "best" clustering method:** K-means scales well but assumes spherical clusters; hierarchical provides interpretable dendrograms but doesn't scale; model-based is flexible but computationally intensive.
4. **Selecting K is part art, part science:** Use elbow plots, silhouette scores, BIC, and domain knowledge together.
5. **Interpret clusters post-hoc:** After clustering, examine cluster means and representative records to assign meaningful labels.
6. **PCA loadings reveal latent structure:** Large loadings indicate which original variables drive each component; use this to name components meaningfully.
7. **Categorical data requires special handling:** One-hot encoding works for few categories; Gower's distance is better for mixed data.
8. **Unsupervised learning feeds supervised learning:** Use cluster assignments as features, PCA components as reduced inputs, or anomaly scores for fraud detection.

---

## Connections to Other Chapters

- **Chapter 1:** EDA tools (boxplots, histograms) help visualise clusters and PCA results; correlation matrices inform PCA interpretation.
- **Chapter 2:** Bootstrap methods can assess cluster stability; sampling distributions underpin model-based clustering inference.
- **Chapter 4:** PCA is the unsupervised analog of regression — both find linear combinations, but PCA has no outcome variable.
- **Chapter 5:** Cluster assignments can serve as features in classification models.
- **Chapter 6:** Random forests provide variable importance that can guide feature selection before clustering; KNN uses similar distance metrics.

---

## Progress Checklist

- [ ] Read complete chapter (pp. 283–326)
- [ ] Apply PCA to numeric dataset; plot scree plot and interpret top loadings
- [ ] Implement K-means with standardised data; use elbow method to select K
- [ ] Perform hierarchical clustering; interpret dendrogram and cut at meaningful height
- [ ] Fit Gaussian mixture model; compare BIC across K and covariance types
- [ ] Handle mixed numeric/categorical data using Gower's distance
- [ ] Evaluate cluster quality using silhouette score and domain validation
- [ ] Use cluster assignments as features in a downstream supervised model
- [ ] Complete `07_unsupervised_learning.ipynb`
- [ ] Solve all exercises in `exercises.ipynb`
- [ ] Experiment with different scaling methods and linkage criteria in `experiments.ipynb`

---


### Questions I Still Have
- When should I prefer hierarchical clustering over K-means for interpretability vs. scalability?
- How can I validate that discovered clusters are stable and not artifacts of random initialisation?
- What's the best practice for handling high-cardinality categorical variables in clustering?
- How do I decide whether to use PCA for dimensionality reduction vs. letting tree-based models handle feature selection?
- When is model-based clustering worth the computational cost over heuristic methods?


---
