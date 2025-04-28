# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as pyo
import plotly.graph_objs as go

from sklearn.cluster import KMeans
import warnings

# Settings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())

# Distribution plots
plt.figure(1, figsize=(15, 6))
for idx, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    plt.subplot(1, 3, idx + 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[col], bins=15, kde=True)
    plt.title(f'Distplot of {col}')
plt.show()

# Pairplot
sns.pairplot(df, vars=['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue="Gender")
plt.show()

# Scatter plot Age vs Spending Score
plt.figure(figsize=(15, 7))
plt.title('Scatter plot of Age vs Spending Score', fontsize=20)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.scatter(df['Age'], df['Spending Score (1-100)'], s=100)
plt.show()

# Clustering Age vs Spending Score
X1 = df[['Age', 'Spending Score (1-100)']].values

# Elbow method
inertia = []
for n in range(1, 15):
    model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                   tol=0.0001, random_state=111, algorithm='elkan')
    model.fit(X1)
    inertia.append(model.inertia_)

plt.figure(figsize=(15, 6))
plt.plot(range(1, 15), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k (Age vs Spending Score)')
plt.show()

# KMeans with 4 clusters
kmeans1 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=111, algorithm='elkan')
labels1 = kmeans1.fit_predict(X1)
centroids1 = kmeans1.cluster_centers_

# Plot clusters
h = 0.02
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15, 7))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(df['Age'], df['Spending Score (1-100)'], c=labels1, s=100)
plt.scatter(centroids1[:, 0], centroids1[:, 1], s=300, c='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Customers (Age vs Spending Score)')
plt.show()

# Clustering Annual Income vs Spending Score
X2 = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Elbow method
inertia = []
for n in range(1, 11):
    model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                   tol=0.0001, random_state=111, algorithm='elkan')
    model.fit(X2)
    inertia.append(model.inertia_)

plt.figure(figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k (Income vs Spending Score)')
plt.show()

# KMeans with 5 clusters
kmeans2 = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=111, algorithm='elkan')
labels2 = kmeans2.fit_predict(X2)
centroids2 = kmeans2.cluster_centers_

# Plot clusters
h = 0.02
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = kmeans2.predict(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)

plt.figure(figsize=(15, 7))
plt.imshow(Z2, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=labels2, s=100)
plt.scatter(centroids2[:, 0], centroids2[:, 1], s=300, c='red', alpha=0.5)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of Customers (Income vs Spending Score)')
plt.show()

# Clustering Age, Annual Income, and Spending Score
X3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Elbow method
inertia = []
for n in range(1, 11):
    model = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                   tol=0.0001, random_state=111, algorithm='elkan')
    model.fit(X3)
    inertia.append(model.inertia_)

plt.figure(figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k (Age, Income, Spending Score)')
plt.show()

# KMeans with 6 clusters
kmeans3 = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, random_state=111, algorithm='elkan')
labels3 = kmeans3.fit_predict(X3)
df['cluster'] = labels3

# 3D plot
trace1 = go.Scatter3d(
    x=df['Age'],
    y=df['Spending Score (1-100)'],
    z=df['Annual Income (k$)'],
    mode='markers',
    marker=dict(
        color=df['cluster'],
        size=10,
        line=dict(
            color=df['cluster'],
            width=12
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    title='Clusters with respect to Age, Income, and Spending Score',
    scene=dict(
        xaxis=dict(title='Age'),
        yaxis=dict(title='Spending Score'),
        zaxis=dict(title='Annual Income')
    )
)

fig = go.Figure(data=data, layout=layout)
pyo.iplot(fig)
