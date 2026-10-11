# -*- coding: utf-8 -*-
"""
Created on Mon May  5 07:38:38 2025

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('C:/360DG/Datasets/iris.csv')

# Rename columns for consistency
df.columns = ['id','sepal_length', 'sepal_width', 
              'petal_length', 'petal_width', 'species']
df.dtypes
#There are no categorical columns so no count plot,pie charts,
# Shape and Size
print("Shape of dataset:", df.shape)
print("Size of dataset:", df.size)
#Inference: Dataset has 150 rows and 6 columns (1 ID, 4 numerical features, and 1 target). It is small and manageable for EDA.

df.describe()
'''
| Element | Meaning                         | Business Use Case                                    |
| --------| --------------------------------| ---------------------------------------------------- |
| Min/Max| Range of measurements            | Helps define product boundaries (e.g., flower sizes) |
| Q1–Q3 | Spread of middle 50% of the data  | Detects variation and 
                                             design tolerance margins       |
| Median| Middle point of distribution      | Useful in targeting 
                                              "typical" customers or scenarios |
| IQR   | Q3 - Q1 (compactness of majority values)| 
                       Important for understanding product consistency      |
'''

# ------------------- Business Moment Decisions -------------------

# 1️First Moment: Mean (Central Tendency)
mean_values = df.mean(numeric_only=True)
print("\nMean (First Moment):\n", mean_values)
# Inference: Mean gives the average measurement.
# Helps in understanding the typical feature size across flowers. 
#Useful for manufacturing parts like robotic arms in floriculture or packaging.

# 2 Second Moment: Variance/Standard Deviation (Dispersion)
var_values = df.var(numeric_only=True)
std_values = df.std(numeric_only=True)
print("\nVariance (Second Moment):\n", var_values)
print("\nStandard Deviation (Second Moment):\n", std_values)
# Inference: Variance shows how widely the feature values 
#are spread. Petal features have much higher variance than 
#sepal features, indicating greater importance for classification.
# Helps in understanding consistency and data reliability.
#  Third Moment: Skewness (Symmetry)
skew_values = df.skew(numeric_only=True)
print("\nSkewness (Third Moment):\n", skew_values)
#  Inference: All features have low skewness, indicating near-normal distribution — good for statistical modeling.
| Feature          | Skewness | Insight (Short)                                                       |
| ---------------- | -------- | --------------------------------------------------------------------- |
| **Sepal Length** | +0.315   | **Slight right skew** → More flowers with **shorter sepal lengths**.  |
| **Sepal Width**  | +0.334   | **Slight right skew** → More flowers with **narrow sepals**.          |
| **Petal Length** | −0.274   | **Slight left skew** → More flowers with **longer petals**.           |
| **Petal Width**  | −0.105   | **Nearly symmetric** → Distribution is **almost normal**, minor skew. |


# Fourth Moment: Kurtosis (Peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print("\nKurtosis (Fourth Moment):\n", kurt_values)
# Inference: Kurtosis near zero shows moderately peaked distribution — suitable for classification with fewer extreme values (outliers).

| Feature          | Kurtosis | Insight (Short)                                                         |
| ---------------- | -------- | ----------------------------------------------------------------------- |
| **Sepal Length** | −0.552   | **Platykurtic** → Flatter distribution, **fewer outliers**.             |
| **Sepal Width**  | +0.291   | **Leptokurtic** → Close to normal, **moderate tail thickness**.          |
| **Petal Length** | −1.402   | **Strongly platykurtic** → Very flat, **very few extreme values**.      |
| **Petal Width**  | −1.340   | **Strongly platykurtic** → Less peaked, **data is spread and uniform**. |

Mesokurtic (Kurtosis ≈ 0)
Normal distribution
Moderate peak and tail thickness.
Example: Standard bell curve (Gaussian distribution).
Leptokurtic (Kurtosis > 0)
Heavy-tailed distribution
High peak and more outliers.
Data is clustered near the mean,
 but has extreme values in tails.
 Platykurtic (Kurtosis < 0)
Flat distribution
Broad peak and fewer outliers.
Tails are lighter; values are more uniformly spread.d

sns.histplot(df.sepal_length,kde=True)
'''
The histogram shows a roughly bell-shaped (unimodal) curve — 
somewhat close to a normal distribution, 
but slightly right-skewed (longer tail on the right).
This means most flowers have moderate sepal lengths, 
with fewer having very short or very long sepals.
'''
sns.histplot(df.sepal_width,kde=True)
'''
The histogram shows a slightly left-skewed (negatively skewed) 
distribution.
Most of the values are concentrated around the middle 
(near 3.0 cm), with fewer samples having very large or 
very small sepal widths.

'''
sns.histplot(df.petal_length,kde=True)
'''
The histogram is bimodal, meaning it has two distinct peaks.
This indicates that petal length values group into two separate 
clusters, likely corresponding to different Iris species.

The first peak is around 1–2 cm.
The second peak is around 4–5 cm.

'''
sns.histplot(df.petal_width,kde=True)
'''
The histogram is bimodal, showing two distinct peaks.
This means there are two groups of flowers with clearly 
different petal widths — corresponding to different 
species of Iris.
The first peak is around 0.2 cm.
The second broader peak is between 1.2 cm and 2.0 cm.

'''
sns.boxplot(df.petal_length)


# Central tendencies
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
# Inference: Median and mode help compare with mean to identify skew. Here, mean ≈ median, confirming symmetric data.

# ------------------- Visual Explorations -------------------

# Histogram of each column
df.drop(columns='id').hist(figsize=(10, 8), color='skyblue', 
                           edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()
#  Inference: 
'''
    Histograms show petal features are more spread out than sepal features. Petal length is a key differentiator.
1. Sepal Length
Distribution: Slightly right-skewed.
Peaks around 5.5 to 6.5 cm.
A wide spread, with some values close to 8 cm 
(mild outliers possible).
Platykurtic behavior → Flat peak and diverse spread.
Insight: Sepal length varies significantly between 
species; useful for classification.

2. Sepal Width
Distribution: Nearly normal but slightly right-skewed.
Most values lie between 2.8 to 3.4 cm.
Mesokurtic → Fairly normal shape with balanced tails.
Insight: Less variation in sepal width across species, may need feature scaling.

3. Petal Length
Distribution: Multimodal (3 distinct peaks).
Clusters visible around 1.5 cm, 4.5 cm, and 5.5+ cm.
Clearly separates species (Setosa, Versicolor, Virginica).
Strongly platykurtic (flat, widely spread distribution).
Insight: Petal length is a highly discriminative feature for species prediction.
4. Petal Width
Distribution: Also multimodal.
Peaks around 0.2, 1.5, and 2 cm.
Large differences between species.
Also platykurtic, indicating distinct clusters and spread.
Insight: Very useful for classification — strongly separates Setosa from the others.
'''
#sns.boxplot(df.petal_width)

# Scatter Plot: Sepal dimensions
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()
#  Inference: Shows overlap among species using sepal dimensions — may not be great for species classification alone.
''' 
Iris-setosa (blue dots):
Clearly forms a separate cluster on the left.
Has shorter sepal length (≈ 4.5–5.5 cm) and wider sepal width (≈ 3.0–4.5 cm).
Distinctly different from the other two species — well-separated.

2. Iris-versicolor (orange dots):
Lies in the mid-range for both length and width.
Sepal length: ≈ 5.0–7.0 cm
Sepal width: ≈ 2.0–3.4 cm
Moderate overlap with Iris-virginica,
 but still forms a rough cluster.

3. Iris-virginica (green dots):
Longest sepal length (≈ 6.0–7.9 cm).
Sepal width is more spread out, often overlapping with versicolor.
Difficult to separate from versicolor using only these two features.
'''

# Joint Plot: Sepal
sns.jointplot(data=df, x="sepal_length", y="sepal_width", kind="scatter")
plt.suptitle("Joint Plot: Sepal Length vs Sepal Width", y=1.02)
plt.show()
'''
#  Inference: Shows bivariate relationship + marginal distributions. Useful to detect concentration of values.
Sepal Length is mostly between 5.0–6.5 cm and is slightly right-skewed.
Sepal Width clusters around 2.8–3.4 cm, close to normal distribution.
The scatter shows a weak positive correlation — longer sepals tend to have slightly wider sepals.
A few outliers exist (very short width or long length).
No clear clustering; these two features alone may not be ideal for classification.

'''
# Joint Plot with species hue
sns.jointplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.suptitle("Joint Plot with Species Hue", y=1.02)
plt.show()
'''
#  Inference: Shows that Setosa is clearly separable, while Versicolor and Virginica have some overlap.
Iris-setosa (blue):
Sepal width: widest and most varied (3.2–4.4 cm).
Sepal length: shortest (~4.5–5.8 cm).
Clearly separated from other species.
Iris-versicolor (orange):
Sepal length: ~5.0–7.0 cm.
Sepal width: more concentrated (~2.3–3.3 cm).
Slight overlap with Iris-virginica.
Iris-virginica (green):
Sepal length: longest (~5.5–8.0+ cm).
Sepal width: similar to versicolor but slightly more scattered.
'''
# Pair Plot
sns.pairplot(df.drop(columns='id'), hue="species")
plt.suptitle("Pair Plot of All Features", y=1.02)
plt.show()
'''
#  Inference: Petal length and width give best class separation — important for model training.
Feature-wise Insights:
Petal Length & Petal Width:
Strong positive correlation for all species.
Best features to classify the species.
Sepal Length & Sepal Width:
Weaker correlation, especially in Iris-setosa.
More overlap between versicolor and virginica.
Distributions:
Petal features are clearly clustered and distinct.
Sepal width of setosa is higher than others; helpful in initial separation.
'''
# Boxplot: Detect outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df.drop(columns=['id']), orient='h')
plt.title("Boxplot of All Features")
plt.show()

'''
#  Inference: No significant outliers. Setosa has distinctly 
#smaller petal sizes — may help early rule-based classification.
1. Sepal Length (Top Blue Box):
Symmetrical distribution.
No outliers.
Median is slightly left-shifted → mild left skew.
Wide IQR (interquartile range) → more spread in values.
2. Sepal Width (Orange Box):
Has visible outliers (small circles).
Median is centered but distribution is a bit left-skewed.
Narrower IQR → less variation than sepal length.
3. Petal Length (Green Box):
Longest IQR → most variable feature.
Slight right skew as median is closer to Q1.
No clear outliers.
4. Petal Width (Red Box):
Slight right skew.
Relatively small range compared to petal length.
No outliers, but compressed distribution.
'''
# Heatmap (Correlation)
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns='id').corr(numeric_only=True), 
            annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
'''
#  Inference: Petal length and petal width have strong positive correlation (0.96). Avoid using both in tree-based models to reduce multicollinearity.
This heatmap shows Pearson correlation coefficients (range from -1 to +1).
Strong positive correlation: values close to +1
Strong negative correlation: values close to –1
Zero correlation: no linear relationship
 Key Observations:
Strong Positive Correlations:
Petal length ↔ Petal width: 0.96
Almost perfect correlation → when petal length increases, petal width increases too.
Sepal length ↔ Petal length: 0.87
Sepal length ↔ Petal width: 0.82
Sepal length is moderately correlated with both petal dimensions → useful for classification.

Negative Correlations:
Sepal width ↔ Petal length: -0.42
Sepal width ↔ Petal width: -0.36
As petal size increases, sepal width
 tends to decrease → weak negative correlation.
Low/Negligible Correlation:
Sepal length ↔ Sepal width: -0.11
No meaningful linear relationship.
Insights for Feature Selection:
Petal length and petal width are 
highly correlated → may lead to multicollinearity.
You can retain either one for simpler models or 
use both for complex models like decision trees or
 neural nets.
Sepal width has low correlation with other features,
 which might help as an independent predictor.
'''
# PDF and CDF for each feature
num_cols = df.select_dtypes(include=np.number).columns.drop('id')

for col in num_cols:
    plt.figure(figsize=(12, 5))

    # PDF
    plt.subplot(1, 2, 1)
    sns.kdeplot(df[col], fill=True, color='blue')
    plt.title(f'PDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')

    # CDF
    plt.subplot(1, 2, 2)
    sorted_vals = np.sort(df[col])
    y_vals = np.arange(len(sorted_vals)) / float(len(sorted_vals))
    plt.plot(sorted_vals, y_vals, marker='.', linestyle='none', color='green')
    plt.title(f'CDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.show()
 '''     
    
# Inference: PDF shows distribution shape; CDF shows how values accumulate. Useful in thresholding or percentile decisions.
PDF (Left Plot) — Probability Density Function of petal_width:
Bimodal Distribution:
The curve shows two peaks, indicating two dominant groups in the data.
This aligns with the Iris dataset,
 which has multiple species with distinct petal widths.
Peaks observed around:
~0.3 (likely corresponds to Iris-setosa)
~1.5 (likely corresponds to versicolor or virginica)
Right-skewed tail:
Some values extend up to 2.5+, though they are less frequent.

CDF (Right Plot) — Cumulative Distribution Function of petal_width:
Smooth upward curve:
Indicates a gradual accumulation of data points.
~60% of data has petal_width < 1.3
Suggests majority of flowers have relatively 
small petal widths.
The step-like structure in the scatter plot shows:
The feature has discrete-like behavior, likely due to repeated values.
'''
'''
 Business or Modeling Insights:
The bimodal nature suggests petal_width is
 a strong discriminator between Iris species.
In classification tasks, 
this feature can significantly contribute 
to model accuracy.
May be useful to perform clustering or threshold-based segmentation using petal width.

Probability Density Function of sepal_width:
Unimodal & Slightly Left-Skewed:
Single peak around 3.0 cm, indicating it’s the most frequent sepal width.
Left tail is longer → some smaller values (as low as ~2.0 cm).
Bell-shaped distribution:
Suggests sepal width follows an approximately 
normal distribution, though slightly skewed.

Cumulative Distribution Function of sepal_width:
Smooth S-curve:
Indicates gradual accumulation; supports the unimodal PDF observation.
~70% of values are below 3.4 cm:
Most flowers have moderate sepal widths.
Flat tails at the start and end:
Fewer extreme small (<2.3) or large (>4.0) widths.

Interpretation for Analysis/Modeling:
sepal_width is less discriminative compared to petal_width due to:
Overlapping values across species.
Normal-like distribution (less separable groups).
Good for EDA or descriptive statistics,
 but may have limited predictive power for classification without combining with other features.

Petal Length Distribution:
Bimodal Shape:
Two distinct peaks:
One around 1.5 cm (Setosa)
Another around 4.5–5.5 cm (Versicolor & Virginica)
Clear separation between groups — ideal for classification.
Highly Informative Feature:
Indicates strong class separability.

CDF (Right Plot) — Cumulative Distribution:
Steep jumps between 1.0–2.0 cm and 4.0–6.0 cm:
Confirms presence of distinct clusters (species).
Flat region between 2.0–4.0 cm:
Very few petal lengths in this range → natural gap between species.
nterpretation:
petal_length is a very powerful predictor for species classification.
Ideal for decision boundaries in models like k-NN, SVM, or Decision Trees.

PDF (Left Plot) — Petal Width Distribution:
Bimodal Distribution:
Two prominent peaks:
One near 0.3 cm (Setosa)
Another around 1.5 cm (Versicolor & Virginica)
Indicates clear separation between species.
Gap between 0.6–1.0 cm:
Few or no samples → useful for threshold-based classification.

CDF (Right Plot) — Cumulative Distribution:
Sharp increase at two ranges:
Around 0.2–0.5 cm and 1.2–1.8 cm → confirms grouping.
Flat mid-range (~0.6–1.0 cm):
Very few samples in this interval → supports distinct clusters.
'''
# Count Plot: Class distribution
sns.countplot(data=df, x='species', palette='Set3')
plt.title("Count of Each Iris Species")
plt.show()

#  Inference: Balanced class distribution — good for unbiased ML model training.

# Bar Graph: Feature means per species
df.groupby('species').mean(numeric_only=True).plot(kind='bar', figsize=(8,6))
plt.title("Mean Feature Values per Species")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

'''
# Inference: Petal length and width vary significantly between species — top features for classification models.
Species-wise Mean Feature Comparison :
Petal Length & Petal Width:
Distinctly increasing from Setosa → Versicolor → Virginica.
These are the most discriminative features 
among species.
Sepal Length:
Gradual increase across species.
Less separation than petals, but still useful.
Sepal Width:
Highest in Setosa, slightly lower in Versicolor & Virginica.
Less consistent for classification.
'''
# Pie Chart
df['species'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, figsize=(6, 6), colors=sns.color_palette('Set3'))
plt.title("Distribution of Iris Species")
plt.ylabel("")
plt.show()
#  Inference: All species are equally represented (33.3% each). Confirms balanced dataset.

# FacetGrid Histogram per species
g = sns.FacetGrid(df, col="species")
g.map_dataframe(sns.histplot, x="sepal_length", kde=True)
g.fig.suptitle("Sepal Length Distribution per Species", y=1.1)
plt.show()
#  Inference: Setosa has a tight, left-skewed 
#sepal length distribution. Versicolor and Virginica 
#have more overlap.

