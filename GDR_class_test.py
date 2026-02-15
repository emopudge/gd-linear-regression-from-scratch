import unittest
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from linreg import GDRegressor


BOSTON = pd.read_csv("housing.csv")


class NetworkTestCase(unittest.TestCase):
    def test_metrics(self):
        X = BOSTON[["RM"]]
        y = BOSTON[["MEDV"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=18)

        max_iter, alpha = GDRegressor.find_optimal_params(X, y)
        print(max_iter, alpha)

        model = GDRegressor(alpha, max_iter)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        custom_r2 = GDRegressor.r_squared(y_test, y_pred)
        custom_rmse = GDRegressor.rmse(y_test, y_pred)

        sklearn_r2 = r2_score(y_test, y_pred)
        sklearn_rmse = root_mean_squared_error(y_test, y_pred)

        print("RMSE:", GDRegressor.rmse(y_test, y_pred))
        print("R^2 Score:", GDRegressor.r_squared(y_test, y_pred))

        self.assertGreaterEqual(custom_r2, 0.49)
        self.assertLessEqual(custom_rmse, 6.45)

        self.assertAlmostEqual(custom_r2, sklearn_r2)
        self.assertAlmostEqual(custom_rmse, sklearn_rmse)
