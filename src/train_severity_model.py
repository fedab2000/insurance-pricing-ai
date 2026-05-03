import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


def train_severity_model():
    df = pd.read_csv("data/insurance_pricing_dataset.csv")

    # Severity model uses only policies with claims
    df_claims = df[df["claim_flag"] == 1].copy()

    X = df_claims.drop(columns=["claim_flag", "claim_amount"])
    y = df_claims["claim_amount"]

    categorical_cols = ["location", "vehicle_type", "coverage_type"]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n=== Severity Model Evaluation ===")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²:", round(r2, 3))

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/severity_model.joblib")

    print("\n✅ Severity model saved to outputs/severity_model.joblib")

    return model


if __name__ == "__main__":
    model = train_severity_model()

    sample = {
        "driver_age": 24,
        "vehicle_age": 3,
        "annual_mileage": 18000,
        "location": "Urban",
        "vehicle_type": "Sedan",
        "coverage_type": "Standard",
        "prior_claims": 1,
        "traffic_violations": 0,
        "policy_tenure": 2,
        "current_premium": 1200
    }

    sample_df = pd.DataFrame([sample])
    severity = model.predict(sample_df)[0]

    print("\nSample predicted claim severity:", round(severity, 2))