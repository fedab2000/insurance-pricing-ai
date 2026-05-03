import joblib
import pandas as pd

# Load models
frequency_model = joblib.load("outputs/frequency_model.joblib")
severity_model = joblib.load("outputs/severity_model.joblib")


def predict_expected_loss(input_data):
    df = pd.DataFrame([input_data])

    # Predict probability of claim
    prob = frequency_model.predict_proba(df)[0][1]

    # Predict severity
    severity = severity_model.predict(df)[0]

    expected_loss = prob * severity

    return {
        "claim_probability": round(prob, 3),
        "predicted_severity": round(severity, 2),
        "expected_loss": round(expected_loss, 2)
    }


if __name__ == "__main__":
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

    result = predict_expected_loss(sample)

    print("\n=== Pricing Output ===")
    print(result)