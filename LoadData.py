from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns

def LoadData(): 
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features and standardize
    X = wine_quality.data.features
    y = wine_quality.data.targets
    original = wine_quality.data.original

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, X,original, y

def remove_outliers():
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features and original data
    X = wine_quality.data.features
    original = wine_quality.data.original

    # Define hardcoded outlier thresholds
    outlier_conditions = [
        (X["residual_sugar"] > 20),
        (X["chlorides"] > 0.2),
        (X["free_sulfur_dioxide"] > 100),
        (X["total_sulfur_dioxide"] > 300),
        (X["density"] > 1.0),
        (X["sulphates"] > 1.1)
    ]

    # Combine all conditions using OR (|) operator to flag outliers
    combined_mask = ~pd.concat(outlier_conditions, axis=1).any(axis=1)

    # Apply the mask to remove outliers
    X_no_outliers = X[combined_mask]

    return X_no_outliers, X, original


