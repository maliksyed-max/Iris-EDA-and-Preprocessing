# -*- coding: utf-8 -*-
"""
Created on Tue Jan  27 19:48:39 2026

@author: HP
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Load the Dataset
df = pd.read_csv("C:\iris.csv")

#Rename column for consistency
df.columns = ['id','sepal_length', 'sepal_width', 'petal_length',
              'petal_width', 'species']
df.dtypes
#There are no categorical columns so no count plot, pie charts
#Shape and Size
print("Shape of dataset:", df.shape)
print("Size of dataset:", df.size)
#Inference: Datset has 150 rows and 6 columns

df.describe()
'''
| Element | Meaning                      |  Business Use case
|---------|------------------------------|---------------------------
| Min/Max | Range of measurements        | Helps define product boundaries
| Q1-Q3   | Spread of middle 50% of data | Detects variation and design tolerance margins
| Median  | Middle point of distribution | Useful in targeting "typical" customers or scenarios
| IQR     |Q3-Q1 (Compactness of majority| Important for uderstanding product consistency
                  values)
'''

#-----------Business Moment Decisions------------------------------
#1 First Moment : Mean (Central Tendency)
mean_values = df.mean(numeric_only=True)
print("\nMean (First Moment):\n", mean_values)
#Inference: Mean gives the average measurement
#Helps in understanding the typical feature size across flowers.
#Useful for manufacturing parts like robotic arms in floriculture or packaging

#2 Second Moment: Variance/Standard Deviation (Dispersion)
var_values = df.var(numeric_only=True)
std_values = df.std(numeric_only=True)
print("\nVariance (Second Moment):\n", var_values)
print("\nStandard Deviation (Second Moment):\n", std_values)
#Inference: Variance shows how widely the feature values are spread. 
#Petal features have much higher variance than sepal features, indicating greater 
#importance for classification.
#Helps in understanding consistency and data reliability.

# 3 Third Moment : Skewness (Symmetry)
skew_values = df.skew(numeric_only=True)
print("\nSkewness (Third Moment):\n", skew_values)
#All features have low skewness
| Feature           | Skewness  |  Insight             |
| ----------------- |-----------|----------------------|             
|**Sepal length**   | +0.315    | **Slight right skew**|
|**Sepal width**    | +0.334    | **Slight right skew**|
|**Petal length**   | -0.274    | **Slight left skew** |
|**Petal width**    | -0.105    | **Nearly symmetric** |

# 4 Fourth Moment: Kurtosis (Peakedness)
kurt_values = df.kurtosis(numeric_only=True)
print("\nKurtosis (Fourth Moment):\n", kurt_values)
#Inference: Kurtosis near zero shows moderately peaked distribution
| Feature           | Kurtosis  |  Insight                               |
| ----------------- |-----------|----------------------------------------|             
|**Sepal length**   | -0.552    | **Platykurtic** -> Flatter Distribution|
|**Sepal width**    | +0.291    | **Mesokurtic** -> Close to normal      |
|**Petal length**   | -1.402    | **Strongly platykurtic** -> Very flat  |
|**Petal width**    | -1.340    | **Strongly platykurtic** -> Less peaked|

Mesokurtic (Kurtosis = 0)
Normal distribution
Moderate peak and tail thickness
Example: Standard bell curve (Gaussian distribution)
Leptokurtic (Kurtosis > 0)
Heavy-tailed distribution
High peak and more outliers.
Data is clustered near the mean,
but has extreme values in tails.
Playkurtic (Kurtosis < 0)
Flat distribution
Broad peak and fewer outliers.
Tails are lighter; values are more uniformly spread

sns.histplot(df.sepal_length, kde=True)
'''
The histogram shows a roughly bell-shaped (unimodal) curve —
somewhat close to a normal distribution,
but slightly right-skewed (longer tail on the right).
This means most flowers have moderate sepal lengths,
with fewer having very short or very long sepals.
'''

sns.histplot(df.sepal_width, kde=True)
'''
The histogram shows a slightly left-skewed (negatively skewed)
distribution.
Most of the values are concentrated around the middle
(near 3.0 cm), with fewer samples having very large or
very small sepal widths.
'''

sns.histplot(df.petal_length, kde=True)
'''
The histogram is bimodal, meaning it has two distinct peaks.
This means there are two groups of flowers with clearly
different petal widths
The first peak is around 0.2 cm.
The second broader peak is between 1.2 cm and 2.0 cm.
'''



sns.boxplot(df.petal_length)

# Central tendencies
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])

# Inference: Median and mode help compare with mean to identify skew.

# ---------------- Visual Explorations ----------------

# Histogram of each column
df.drop(columns=['id']).hist(figsize=(10, 8), color='skyblue',
                              edgecolor='black')
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

'''
Inference:

Histograms show petal features are more spread out than sepal features.
Sepal Length:
Distribution: Slightly right-skewed.
Peaks around 5.5 to 6.5 cm.
A wide spread, with some values close to 8 cm
(mild outliers possible).
Platykurtic behavior → flat peak and diverse spread.
Insight: Sepal length varies significantly between species; useful for classification.
'''

# Scatter Plot: Sepal dimensions
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()

'''
Inference: Shows overlap among species using sepal dimensions — may not be perfectly separable.

Iris-setosa (blue dots):
Clearly forms a separate cluster on the left.
Has shorter sepal length (~4.5–5.5 cm) and wider sepal width (~3.0–4.5 cm).
Distinctly different from the other two species — well-separated.

Iris-versicolor (orange dots):
Lies in mid-range for both length and width.
Sepal Length: ~5.0–7.0 cm
Sepal Width: ~2.6–3.4 cm
Moderate overlap with Iris-virginica
but still forms a rough cluster.

Iris-virginica (green dots):
Longest sepal length (~6.0–7.9 cm).
Sepal width is more spread out, often overlapping with versicolor.
Difficult to separate from versicolor using only these two features.
'''

# Joint Plot: Sepal
sns.jointplot(data=df, x="sepal_length", y="sepal_width", kind="scatter")
plt.suptitle("Joint Plot: Sepal Length vs Sepal Width", y=1.02)
plt.show()

'''
Inference: Shows bivariate relationship + marginal distributions.
Sepal Length is mostly between 5.0–6.5 cm and slightly right-skewed.
Sepal Width clusters around 2.8–3.4 cm, close to normal distribution.
The scatter shows weak positive correlation — longer sepals tend to be slightly wider.
A few outliers exist (very short width or long length).
No clear clustering; these two features alone may not be ideal for classification.
'''

# Joint Plot with species hue
sns.jointplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.suptitle("Joint Plot with Species Hue", y=1.02)
plt.show()

'''
Inference: Shows that Setosa is clearly separable, while Versicolor and Virginica overlap.

Iris-setosa (blue):
Sepal width: widest and most varied (~3.2–4.4 cm).
Sepal Length: shortest (~4.5–5.8 cm).
Clearly separated from other species.

Iris-versicolor (orange):
Sepal Length: ~5.0–7.0 cm.
Sepal Width: more concentrated (~2.3–3.3 cm).
Slight overlap with Iris-virginica.

Iris-virginica (green):
Sepal Length: longest (~5.5–8.0+ cm).
Sepal Width: similar to versicolor but slightly more scattered.
'''

# Pair Plot
sns.pairplot(df.drop(columns=['id']), hue="species")
plt.suptitle("Pair Plot of ALL Features", y=1.02)
plt.show()

'''
Inference: Petal Length and Width give best class separation — important features.

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
plt.title("Boxplot of ALL Features")
plt.show()

'''
Inference: No significant outliers. Setosa has distinctly smaller petal sizes — may help early rule-based classification.

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
Slight right skew.
Relatively small range compared to sepal length.
No outliers, but compressed distribution.
'''

# Heatmap (Correlation)
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns=['id']).corr(numeric_only=True),
            annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

'''
Inference: Petal Length and Petal Width have strong positive correlation.
This heatmap shows Pearson correlation coefficients (range from -1 to +1).

Strong positive correlation: values close to +1
Strong negative correlation: values close to -1
Zero correlation: no linear relationship

Key Observations:
Petal Length ↔ Petal Width: 0.96
Almost perfect correlation — when petal length increases, petal width also increases.

Sepal Length ↔ Petal Length: 0.87
Sepal Length ↔ Petal Width: 0.82

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
    plt.plot(sorted_vals, y_vals, color='red')
    plt.title(f'CDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')

    plt.tight_layout()
    plt.show()

'''
 Inference: PDF shows distribution shape; CDF shows how values accumulate. Useful for EDA (Left) — Probability Density Function of petal_width:
Bimodal Distribution:
The curve shows two peaks, indicating two dominant groups in the data.
This aligns with the Iris dataset,
which has multiple species with distinct petal widths.
Peaks observed around:
~0.3 (likely corresponds to Iris-setosa)
~1.5 (likely corresponds to versicolor or virginica)
Right-skewed tail:
Some values extend up to 2.5+, though they are less frequent.

# CDF (Right Plot) — Cumulative Distribution Function of petal_width:
Smooth upward curve:
Indicates a gradual accumulation of data points.
~60% of data has petal_width < 1.3
Suggests majority of flowers have relatively small petal widths.
The step-like structure in the scatter plot shows:
The feature has discrete-like behavior, likely due to repeated values.

# Business or Modeling Insights:
The bimodal nature suggests petal_width is
a strong discriminator between Iris species.
In classification tasks,
this feature can significantly contribute
to model accuracy.
May be useful to perform clustering or threshold-based segmentation using petal_width.

# Probability Density Function of sepal_width:
Unimodal & Slightly Left-Skewed:
Single peak around 3.0 cm, indicating it's the most frequent sepal width.
Left tail is longer → some smaller values (as low as ~2.0 cm).
Bell-shaped distribution:
Suggests sepal_width follows an approximately
normal distribution, though slightly skewed.

# Cumulative Distribution Function of sepal_width:
Smooth S-curve:
Indicates gradual accumulation; supports the unimodal PDF observation.
~70% of values are below 3.4 cm:
Most flowers have moderate sepal widths.
Flat tails at the start and end:
Fewer extreme small (<2.3) or large (>4.0) widths.

# Interpretation for Analysis/Modeling:
sepal_width is less discriminative compared to petal_width due to:
Overlapping values across species.
Normal-like distribution (Less separable groups).
Good for EDA or descriptive statistics,
but may have limited predictive power for classification without combining with other features.

# Petal Length Distribution:
Bimodal Shape:
Two distinct peaks:
One around 1.5 cm (Setosa)
Another around 4.5–5.5 cm (Versicolor & Virginica)
Clear separation between groups — ideal for classification.
Highly Informative Feature:
Indicates strong class separability.

# CDF (Right Plot) — Cumulative Distribution:
Steep jumps between 1.0–2.0 cm and 4.0–6.0 cm:
Confirms presence of distinct clusters (species).
Very few petal lengths in this range → natural gap between species.
Interpretation:
petal_length is a very powerful predictor for species classification.
Ideal for decision boundaries in models like k-NN, SVM, or Decision Trees.

# PDF (Left Plot) — Petal Width Distribution:
Bimodal Distribution:
Two prominent peaks:
One near 0.3 cm (Setosa)
Another around 1.5 cm (Versicolor & Virginica)
Indicates clear separation between species.
Gap between 0.6–1.0 cm:
Few or no samples → useful for threshold-based classification.

# CDF (Right Plot) — Cumulative Distribution:
Sharp increase at two ranges:
Around 0.2–0.5 cm and 1.2–1.8 cm → confirms grouping.
Flat mid-range (~0.6–1.0 cm):
Very few samples in this interval → supports distinct clusters.
'''

# Count Plot: Class distribution
sns.countplot(data=df, x='species', palette='Set3')
plt.title("Count of Each Iris Species")
plt.show()

# Inference: Balanced class distribution — good for unbiased ML model training.

# Bar Graph: Feature means per species
df.groupby('species').mean(numeric_only=True).plot(kind='bar', figsize=(8,6))
plt.title("Mean Feature Values per Species")
plt.ylabel("Mean Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# ============================
# Inference:
# ============================

# Petal Length and width vary significantly between species – top feature.
# Species-wise Mean Feature Comparison:

# Petal Length:
# Steadily increasing from Setosa → Versicolor → Virginica.
# Distinctly increases.
# These are the most discriminative features among species.

# Sepal Length:
# Gradual increase across species.
# Moderate separation.

# Sepal Width:
# Less separation than petals, but still useful.
# Highest in Setosa, slightly lower in Versicolor & Virginica.
# Shows consistency for classification.


# ============================
# Pie Chart – Species Distribution
# ============================

df['species'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    figsize=(6,6)
    
)

plt.title("Distribution of Iris Species")
plt.ylabel("")
plt.show()