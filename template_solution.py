import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct, RationalQuadratic, WhiteKernel, RBF, ExpSineSquared, ConstantKernel, Matern)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

#Gemini and ChatGPT where used for kernal combination ideas. Testing was done using cross-fold on training data
#and uploading various combinations

def load_data():
    # Load raw train and test files
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print("\n")

    print("Test data:")
    print("Shape:", test_df.shape)
    print(test_df.head(2))
    print("\n")

    # Convert the categorical season column into numeric dummy variables
    train_df = pd.get_dummies(train_df, columns=["season"], dtype=float)
    test_df = pd.get_dummies(test_df, columns=["season"], dtype=float)

    # Ensure train and test have the same feature columns
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Only impute input features, never the target
    impute_cols = [col for col in train_df.columns if col != "price_CHF"]

    # Use interpolation first because neighboring rows may contain related time information
    train_df[impute_cols] = train_df[impute_cols].interpolate(
        method="linear", limit_direction="both"
    )
    test_df[impute_cols] = test_df[impute_cols].interpolate(
        method="linear", limit_direction="both"
    )

    # Iterative imputation fills any remaining missing feature values
    imputer = IterativeImputer(random_state=42)
    train_df[impute_cols] = imputer.fit_transform(train_df[impute_cols])
    test_df[impute_cols] = imputer.transform(test_df[impute_cols])

    # Add short-term smoothed versions of numeric features
    rolling_cols = [col for col in impute_cols if not col.startswith("season_")]

    for col in rolling_cols:
        train_df[f"{col}_rolling_3"] = train_df[col].rolling(window=3, min_periods=1).mean()
        test_df[f"{col}_rolling_3"] = test_df[col].rolling(window=3, min_periods=1).mean()

    print("Remaining NaNs in training data:")
    print(train_df.isna().sum())
    print("\n")

    print("Remaining NaNs in test data:")
    print(test_df.isna().sum())
    print("\n")

    # Rows without a target value cannot be used for supervised training
    train_df = train_df.dropna(subset=["price_CHF"])

    # Separate inputs and target
    X_train_df = train_df.drop("price_CHF", axis=1)
    y_train = train_df["price_CHF"].values
    X_test_df = test_df.drop("price_CHF", axis=1, errors="ignore")

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Standardization is important for Gaussian Processes because kernels depend on distances
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    assert (X_train.shape[1] == X_test.shape[1]) and \
           (X_train.shape[0] == y_train.shape[0]) and \
           (X_test.shape[0] == 100), "Invalid data shape"

    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._gpr = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._x_train = X_train
        self._y_train = y_train

        # Combine:
        # - a linear component (DotProduct)
        # - a flexible nonlinear component (Matern)
        # - a noise term (WhiteKernel)

        num_features = self._x_train.shape[1]
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e2))
            + ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(num_features),
                length_scale_bounds=(1e-3, 1e3),
                nu=1.5,
            )
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )

        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            random_state=42,
        )
        self._gpr.fit(self._x_train, self._y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self._gpr.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()

    model = Model()
    model.fit(X_train=X_train, y_train=y_train)

    y_pred = model.predict(X_test)

    dt = pd.DataFrame(y_pred, columns=["price_CHF"])
    dt.to_csv("results.csv", index=False)

    print("\nResults file successfully generated!")