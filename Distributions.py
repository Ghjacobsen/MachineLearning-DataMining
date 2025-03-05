from LoadData import LoadData, remove_outliers
import matplotlib.pyplot as plt
import seaborn as sns

X_scaled, X, original, y = LoadData()
X_no_outliers, X, original = remove_outliers()


# Select specific features to plot
selected_features = ["pH", "sulphates", "residual_sugar"]

# Create subplots for the selected features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # One row, three columns

for i, feature in enumerate(selected_features):
    sns.kdeplot(X[feature], ax=axes[i], fill=True)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel("Feature Values")
    axes[i].set_ylabel("Density")

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("smooth_distributions.png")
plt.show()
