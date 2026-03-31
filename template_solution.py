import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RationalQuadratic, WhiteKernel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data():
    """
    Loads training and test data, imputes missing values using
    learned patterns from the training set, and returns
    X_train, y_train, X_test.
    """

    # Load training data
    train_df = pd.read_csv("train.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print("\n")

    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print("Shape:", test_df.shape)
    print(test_df.head(2))
    print("\n")

    # One-hot encode season
    train_df = pd.get_dummies(train_df, columns=["season"], dtype=float)
    test_df = pd.get_dummies(test_df, columns=["season"], dtype=float)

    # Align columns between train and test
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Exclude target from imputation
    impute_cols = [col for col in train_df.columns if col != "price_CHF"]

    # Fit imputer on training data only
    imputer = IterativeImputer(random_state=42)
    train_df[impute_cols] = imputer.fit_transform(train_df[impute_cols])
    test_df[impute_cols] = imputer.transform(test_df[impute_cols])

    print("Remaining NaNs in training data:")
    print(train_df.isna().sum())
    print("\n")

    print("Remaining NaNs in test data:")
    print(test_df.isna().sum())
    print("\n")

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["price_CHF"])

    # Split into features and target
    X_train_df = train_df.drop("price_CHF", axis=1)
    y_train = train_df["price_CHF"].values

    # Test features
    X_test_df = test_df.drop("price_CHF", axis=1, errors="ignore")

    # Convert to numpy
    X_train = X_train_df.values
    X_test = X_test_df.values

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

        # Best kernel: Dot + RQ + White
        kernel = DotProduct() + RationalQuadratic() + WhiteKernel()

        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=42
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