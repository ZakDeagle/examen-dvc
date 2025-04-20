import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

best_params = joblib.load("models/best_params.pkl")

model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/trained_model.pkl")
