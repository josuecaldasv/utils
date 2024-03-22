from sklearn.base import BaseEstimator, RegressorMixin
from scipy.interpolate import UnivariateSpline
import pandas as pd

class SplineRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper class for UnivariateSpline to make it compatible with scikit-learn's API.

    Objectives:
    - To provide a spline regression model that can be used within the scikit-learn ecosystem.
    - To handle duplicate values in the 'X' dataset by averaging corresponding 'y' values before fitting the spline.
    
    Input:
    - s: float, smoothing factor passed to UnivariateSpline. 
         A value of 0 means no smoothing (interpolating spline through all data points).
    - k: int, degree of the spline. Must be <= 5. Default is k=3, a cubic spline.
    
    Methods:
    - fit: Fits the model to the data. X is expected to be a 2D array-like structure, and y is 1D.
    - predict: Predicts the target values using the fitted spline model. X is expected to be a 2D array-like structure.
    - promediar_duplicados: A static method used to average 'y' values for duplicate 'x' values in the dataset.

    Output:
    - The class does not directly output but provides 'fit' and 'predict' methods for model training and prediction.
    
    Usage:
    ```python
    model = SplineRegressor(s=0, k=3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ```
    
    Note: The class uses a static method 'promediar_duplicados' to preprocess the data and average 'y' values for 
    duplicate 'x' values, ensuring that the spline is fitted to a dataset with unique 'x' values.
    """
    
    def __init__(self, s=0, k=3):
        self.s = s
        self.k = k
        self.model = None
    
    def fit(self, X, y):
        """
        Fits the spline model to the data, handling duplicates in 'X' by averaging corresponding 'y' values.

        Parameters:
        - X: array-like, shape (n_samples,)
            The input variables.
        - y: array-like, shape (n_samples,)
            The target values.
        
        Returns:
        - self: returns an instance of self.
        """
        X, y = self.promediar_duplicados(X.ravel(), y)
        self.model = UnivariateSpline(X, y, s=self.s, k=self.k)
        return self
    
    def predict(self, X):
        """
        Predicts the target values using the fitted spline model.

        Parameters:
        - X: array-like, shape (n_samples,)
            The input variables.
        
        Returns:
        - y_pred: array, shape (n_samples,)
            The predicted values.
        """
        if self.model is not None:
            return self.model(X.ravel())
        else:
            raise ValueError("This model is not fitted yet.")
    
    @staticmethod
    def promediar_duplicados(X, y):
        """
        Averages 'y' values for duplicate 'x' values in the dataset.

        Parameters:
        - X: array-like, shape (n_samples,)
            The input variables.
        - y: array-like, shape (n_samples,)
            The target values.
        
        Returns:
        - X_unique: array, shape (n_unique_samples,)
            The unique input variables.
        - y_avg: array, shape (n_unique_samples,)
            The averaged target values for each unique input variable.
        """
        df = pd.DataFrame({"X": X, "y": y}).groupby("X", as_index=False).mean()
        return df["X"].values, df["y"].values
