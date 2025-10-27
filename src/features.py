import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(df: pd.DataFrame, keep_var=0.95):
    X = df.drop(columns=["label"])
    pca = PCA(n_components=keep_var, svd_solver="full")
    Xr = pca.fit_transform(X)
    Xr_df = pd.DataFrame(Xr, index=df.index)
    Xr_df["label"] = df["label"].values
    return Xr_df, pca
