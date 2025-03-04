from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler
import pandas as pd

def LoadData():
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Extract features and standardize
    X = wine_quality.data.features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, X
