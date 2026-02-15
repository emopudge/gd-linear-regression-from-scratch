import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd


class GDRegressor:
    def __init__(self, alpha=0.001, n_iter=100, progress=True):
        self.alpha = alpha
        self.n_iter = int(n_iter)
        self.progress = progress
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self.theta_history = []

    def fit(self, X_train, y_train):
        """
        Trains the model using gradient descent.
        """

        # Convert to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X = X_train.values
        elif isinstance(X_train, np.ndarray):
            X = X_train
        else:
            raise TypeError("Unsupported type for X_train")

        if isinstance(y_train, pd.Series):
            y = y_train.values.ravel()
        elif isinstance(y_train, pd.DataFrame):
            y = y_train.values.ravel()
        elif isinstance(y_train, np.ndarray):
            y = y_train.ravel()
        else:
            raise TypeError("Unsupported type for y_train")

        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])  # Add intercept term

        self.theta = np.zeros(X.shape[1])
        self.loss_history = []
        self.theta_history = []

        for i in range(self.n_iter):
            y_pred = X @ self.theta
            error = y_pred - y

            gradient = (1 / m) * X.T @ error

            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                print(f"[ERROR] Gradient contains NaN/inf at iteration {i}.")
                break

            self.theta -= self.alpha * gradient

            loss = (1 / (2 * m)) * np.dot(error, error)
            self.loss_history.append(loss)
            self.theta_history.append(self.theta.copy())

            if self.progress and i % (self.n_iter // 10) == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    def predict(self, X_test):
        """
        Predicts target values based on trained model.
        """

        if isinstance(X_test, pd.DataFrame):
            X = X_test.values
        elif isinstance(X_test, np.ndarray):
            X = X_test
        else:
            raise TypeError("Unsupported type for X_test")

        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])

        predictions = X @ self.theta
        return predictions

    @staticmethod
    def plot_coeffs(model):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(list(range(model.n_iter)), [theta[0] for theta in model.theta_history], label="Intercept (theta_0)")
        plt.title("Intercept over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Value")

        plt.subplot(1, 2, 2)
        plt.plot(
            list(range(model.n_iter)),
            [theta[1] for theta in model.theta_history],
            label="Coefficient (theta_1)",
            color="orange",
        )
        plt.title("Coefficient over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_loss_function(model):
        plt.figure(figsize=(8, 5))
        plt.plot(model.loss_history)
        plt.title("Loss Function over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.show()

    @staticmethod
    def z_scaler(feature):
        mean = np.mean(feature)
        std = np.std(feature)
        if std == 0:
            return feature - mean
        return (feature - mean) / std

    @staticmethod
    def rmse(y, y_hat):

        if isinstance(y_hat, (pd.DataFrame, pd.Series)):
            y_hat = y_hat.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        y_hat = y_hat.ravel()
        y = y.ravel()

        result = np.sqrt(np.mean((y_hat - y) ** 2))
        return result

    @staticmethod
    def r_squared(y, y_hat):

        if isinstance(y_hat, (pd.DataFrame, pd.Series)):
            y_hat = y_hat.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        y_hat = y_hat.ravel()
        y = y.ravel()

        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            result = float("-inf")
        else:
            result = 1 - (ss_res / ss_tot)

        return result

    @staticmethod
    def find_optimal_params(X, y):

        best_rmse, best_r2 = np.inf, np.inf

        best_alpha = 0.01
        best_n_iter = 2000

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=18)

        alphas = [0.001, 0.005, 0.01, 0.02, 0.04, 0.1]
        iters = [100, 500, 1000, 2000, 3000]

        for alpha in alphas:
            for n_iter in iters:
                model = GDRegressor(alpha=alpha, n_iter=n_iter, progress=False)
                model.fit(X_train.values, y_train.values)
                y_pred = model.predict(X_test.values)

                current_rmse = GDRegressor.rmse(y_test.values, y_pred)
                current_r2 = GDRegressor.r_squared(y_test.values, y_pred)

                if current_rmse < best_rmse and current_r2 >= 0.49:
                    print(f"[INFO] Suitable parameters found: alpha={alpha}, n_iter={n_iter}")
                    return n_iter, alpha

        print("[WARNING] Could not find optimal params. Using defaults.")
        return best_n_iter, best_alpha
