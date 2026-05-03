import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_glm_frequency_model():
    df = pd.read_csv("data/insurance_pricing_dataset.csv")

    X = df.drop(columns=["claim_flag", "claim_amount"])
    y = df["claim_flag"]

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
        ("glm_poisson", PoissonRegressor(
            alpha=0.1,
            max_iter=1000
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    print("\n=== GLM Poisson Frequency Model ===")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("Average predicted claim frequency:", round(predictions.mean(), 4))
    print("Actual claim frequency:", round(y_test.mean(), 4))

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/glm_frequency_model.joblib")

    print("\n✅ GLM frequency model saved to outputs/glm_frequency_model.joblib")

    return model


if __name__ == "__main__":
    train_glm_frequency_model()