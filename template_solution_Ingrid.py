import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic, WhiteKernel


def load_data():
    """
    Loads training and test data, imputes missing values using IterativeImputer,
    and returns X_train, y_train, X_test.
    """

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print()

    print("Test data:")
    print("Shape:", test_df.shape)
    print(test_df.head(2))
    print()

    # One-hot encode season
    train_df = pd.get_dummies(train_df, columns=["season"], dtype=float)
    test_df = pd.get_dummies(test_df, columns=["season"], dtype=float)

    # Align columns so train/test match
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Columns to impute: everything except target
    impute_cols = [col for col in train_df.columns if col != "price_CHF"]

    # Fit imputer on training data only
    imputer = IterativeImputer(random_state=42)
    train_df[impute_cols] = imputer.fit_transform(train_df[impute_cols])
    test_df[impute_cols] = imputer.transform(test_df[impute_cols])

    print("Remaining NaNs in training data:")
    print(train_df.isna().sum())
    print()

    print("Remaining NaNs in test data:")
    print(test_df.isna().sum())
    print()

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["price_CHF"])

    # Split into X/y
    X_train_df = train_df.drop("price_CHF", axis=1)
    y_train = train_df["price_CHF"].values

    # Test does not have price_CHF, but ignore just in case
    X_test_df = test_df.drop("price_CHF", axis=1, errors="ignore")

    X_train = X_train_df.values
    X_test = X_test_df.values

    assert (
        X_train.shape[1] == X_test.shape[1]
        and X_train.shape[0] == y_train.shape[0]
        and X_test.shape[0] == 100
    ), "Invalid data shape"

    return X_train, y_train, X_test


class Model:
    def __init__(self):
        self.model = None
        self.best_kernel_name = None
        self.best_kernel = None

    def _choose_best_kernel(self, X_train: np.ndarray, y_train: np.ndarray):
        kernels = {
            "RBF": RBF(),
            "RQ": RationalQuadratic(),
            "Dot + RBF": DotProduct() + RBF(),
            "Dot + RQ": DotProduct() + RationalQuadratic(),
            "RBF + White": RBF() + WhiteKernel(),
            "RQ + White": RationalQuadratic() + WhiteKernel(),
            "Dot + RBF + White": DotProduct() + RBF() + WhiteKernel(),
            "Dot + RQ + White": DotProduct() + RationalQuadratic() + WhiteKernel(),
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        best_score = float("inf")
        best_name = None
        best_kernel = None

        print("Comparing kernels with 5-fold CV...\n")

        for name, kernel in kernels.items():
            pipeline = make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=kernel,
                    normalize_y=True,
                    random_state=42
                )
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error"
            )

            mean_rmse = -scores.mean()
            std_rmse = scores.std()

            print(f"{name:20s} RMSE = {mean_rmse:.4f}")

            if mean_rmse < best_score:
                best_score = mean_rmse
                best_name = name
                best_kernel = kernel

        print(f"\nBest kernel: {best_name} with RMSE = {best_score:.4f}\n")
        return best_name, best_kernel

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.best_kernel_name, self.best_kernel = self._choose_best_kernel(X_train, y_train)

        # Fit final model on all training data
        self.model = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=self.best_kernel,
                normalize_y=True,
                random_state=42
            )
        )

        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid prediction shape"
        return y_pred


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()

    model = Model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dt = pd.DataFrame({"price_CHF": y_pred})
    dt.to_csv("results.csv", index=False)

    print("Results file successfully generated!")