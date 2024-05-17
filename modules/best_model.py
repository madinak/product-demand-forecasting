from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import joblib


class XGBTrainer:
    def __init__(self, checkpoint_path=None):
        self.model = XGBRegressor(random_state=42)
        self.param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 6, 7],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],  
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0]  
        }
        self.checkpoint_path = checkpoint_path

    def train(self, X_train, y_train, filename='xgb.joblib'):
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid,
                                   cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        # Save best model checkpoint
        if self.checkpoint_path:
            joblib.dump(grid_search.best_estimator_, self.checkpoint_path)
        
        # Save the best model along with its best parameters
        best_model_filename = f"{filename.split('.')[0]}_best_model.joblib"
        joblib.dump((grid_search.best_estimator_, grid_search.best_params_), best_model_filename)
        # Update model attribute with best estimator
        self.model = grid_search.best_estimator_
        # Return best model, best parameters, and best RMSE
        return self.model, grid_search.best_params_, np.sqrt(-grid_search.best_score_)