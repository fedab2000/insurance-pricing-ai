import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_frequency_model():
    df = pd.read_csv("data/insurance_pricing_dataset.csv")

    X = df.drop(columns=["claim_flag", "claim_amount", "current_premium"])
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
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
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
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n=== Frequency Model Evaluation ===")
    print(classification_report(y_test, predictions, zero_division=0))
    print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 3))

    # Feature importance
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": coefficients
    }).sort_values(by="importance", ascending=False)

    print("\n=== Top Risk Drivers ===")
    print(importance_df.head(10))

    print("\n=== Top Protective Factors ===")
    print(importance_df.tail(10))

    os.makedirs("outputs", exist_ok=True)

    importance_df.to_csv(
        "outputs/frequency_feature_importance.csv",
        index=False
    )

    joblib.dump(model, "outputs/frequency_model.joblib")

    print("\n✅ Frequency model saved to outputs/frequency_model.joblib")
    print("✅ Feature importance saved to outputs/frequency_feature_importance.csv")

    return model


if __name__ == "__main__":
    model = train_frequency_model()

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
    prob = model.predict_proba(sample_df)[0][1]

    print("\nSample prediction (probability of claim):", round(prob, 3))