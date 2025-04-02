import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from LoadData import LoadData

X_scaled, X, y_quality, original = LoadData()

y_quality = np.array(y_quality).flatten()

# Perform PCA
pca = PCA(n_components=len(X_scaled.columns))
X_pca = pca.fit_transform(X_scaled)

# 2. Explained Variance Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(X_scaled.columns) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by Number of PCA Components")
plt.grid(True)
plt.savefig("explained_variance.png")
#plt.show() 

# Select only the first 5 PCA components for plotting
num_components_to_plot = 5
components_df = pd.DataFrame(
    pca.components_[:num_components_to_plot],  # Take only the first 5 rows
    columns=X_scaled.columns,
    index=[f"PC{i+1}" for i in range(num_components_to_plot)]
)

# 1. Plot the First 5 Principal Components in Terms of Attributes
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(components_df, cmap="coolwarm", center=0, annot=True, fmt=".2f", xticklabels=X_scaled.columns, yticklabels=components_df.index)
ax.set_title("PCA Component Loadings (First 5 PCs)")
plt.xlabel("Original Features")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Principal Components")
plt.tight_layout()

# Save the figure
plt.savefig("pca_loadings.png", bbox_inches="tight") 



