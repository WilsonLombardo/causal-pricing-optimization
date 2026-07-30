import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

class TLearner:
    def __init__(self):
        # Usamos LightGBM como pide la vacante
        self.model_control = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        self.model_treated = lgb.LGBMRegressor(n_estimators=100, random_state=42)

    def fit(self, X: pd.DataFrame, T: pd.Series, y: pd.Series):
        # Separar datos entre grupo de control (T=0) y tratamiento (T=1)
        X_control, y_control = X[T == 0], y[T == 0]
        X_treated, y_treated = X[T == 1], y[T == 1]

        # Entrenar ambos modelos
        self.model_control.fit(X_control, y_control)
        self.model_treated.fit(X_treated, y_treated)

    def predict_cate(self, X: pd.DataFrame) -> pd.Series:
        # Predecir contrafactuales
        y_pred_control = self.model_control.predict(X)
        y_pred_treated = self.model_treated.predict(X)
        
        # Conditional Average Treatment Effect (CATE)
        cate = y_pred_treated - y_pred_control
        return pd.Series(cate, index=X.index)