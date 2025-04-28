import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = pd.read_csv("IRIS.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values

# Info and head
iris.info()
print(iris.head(10))

# Frequency distribution of species
iris_outcome = pd.crosstab(index=iris["species"], columns="count")
print(iris_outcome)

# Subsets
iris_setosa = iris.loc[iris["species"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["species"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["species"] == "Iris-versicolor"]

# Distribution plots
sns.FacetGrid(iris, hue="species", height=3).map(sns.kdeplot, "petal_length").add_legend()
plt.show()

sns.FacetGrid(iris, hue="species", height=3).map(sns.kdeplot, "petal_width").add_legend()
plt.show()

sns.FacetGrid(iris, hue="species", height=3).map(sns.kdeplot, "sepal_length").add_legend()
plt.show()

# Boxplot
sns.boxplot(x="species", y="petal_length", data=iris)
plt.show()

# Violinplot
sns.violinplot(x="species", y="petal_length", data=iris)
plt.show()

# Pairplot
sns.set_style("whitegrid")
sns.pairplot(iris, hue="species", height=3)
plt.show()
