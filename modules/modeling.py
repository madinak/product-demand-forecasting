from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class DemandPredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'GradientBoosting': GradientBoostingRegressor()
        }

    def train(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)  # Fit the model
            trained_models[name] = model
        return trained_models

    def predict(self, trained_models, X_test, y_test):
        predictions = {}
        scores = {}
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)  # Predict using each trained model
            predictions[name] = y_pred
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            scores[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R^2': r2}
        return predictions, scores

    def get_scores(self, X_test, y_test):
        _, scores = self.predict(self.models, X_test, y_test)
        return scores