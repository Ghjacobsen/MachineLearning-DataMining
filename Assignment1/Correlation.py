from LoadData import LoadData
import matplotlib.pyplot as plt
import seaborn as sns

X_scaled, X, original, y = LoadData()

corr_matrix = X.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig("correlation_heatmap.png")