import matplotlib.pyplot as plt
from LoadData import LoadData

X_scaled, X = LoadData()
# Create subplots for original and standardized boxplots
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Original feature distribution
X.boxplot(ax=axes[0], rot=45, grid=False, showfliers=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
axes[0].set_title("Original Boxplots of Wine Quality Features")
axes[0].set_ylabel("Feature Values")
axes[0].set_xlabel("Features")

# Standardized feature distribution
X_scaled.boxplot(ax=axes[1], rot=45, grid=False, showfliers=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
axes[1].set_title("Standardized Boxplots of Wine Quality Features")
axes[1].set_ylabel("Standardized Values")
axes[1].set_xlabel("Features")

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# metadata 
#print(wine_quality.metadata) 
  
# variable information 
#print(wine_quality.variables) 


