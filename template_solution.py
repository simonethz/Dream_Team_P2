# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

import pandas as pd
import numpy as np

def load_data():
    """
    Loads training and test data, imputes missing values using
    season-wise means computed from the training set, and returns
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

    # Input feature columns only
    feature_cols = [col for col in train_df.columns if col not in ["season", "price_CHF"]]

    # Compute season-wise means from training data
    season_means = train_df.groupby("season")[feature_cols].mean()

    # Fill missing values in training features
    for col in feature_cols:
        train_df[col] = train_df.groupby("season")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    # Fill missing values in test features using training season means
    for col in feature_cols:
        test_df[col] = test_df.apply(
            lambda row: season_means.loc[row["season"], col]
            if pd.isna(row[col]) else row[col],
            axis=1
        )

    print("Remaining NaNs in training data:")
    print(train_df.isna().sum())
    print("\n")

    print("Remaining NaNs in test data:")
    print(test_df.isna().sum())
    print("\n")

    # Drop rows with missing target
    train_df = train_df.dropna(subset=["price_CHF"])

    # One-hot encode season
    train_df = pd.get_dummies(train_df, columns=["season"])
    test_df = pd.get_dummies(test_df, columns=["season"])

    # Split train into features and target
    X_train_df = train_df.drop("price_CHF", axis=1)
    y_train = train_df["price_CHF"].values

    # Align train/test feature columns
    X_train_df, X_test_df = X_train_df.align(test_df, join="left", axis=1, fill_value=0)

    # Convert to numpy arrays
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

        kernel = DotProduct() + RBF() + RationalQuadratic()
        self._gpr = GaussianProcessRegressor(kernel=kernel)

        #self._gpr = GaussianProcessRegressor(kernel=DotProduct())
        self._gpr.fit(self._x_train, self._y_train)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._gpr.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

