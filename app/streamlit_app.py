import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="P&C Pricing AI App", layout="wide")

frequency_model = joblib.load("outputs/frequency_model.joblib")
severity_model = joblib.load("outputs/severity_model.joblib")
feature_importance = pd.read_csv("outputs/frequency_feature_importance.csv")


def calculate_pricing(input_data):
    df = pd.DataFrame([input_data])

    claim_probability = frequency_model.predict_proba(df)[0][1]
    predicted_severity = severity_model.predict(df)[0]
    expected_loss = claim_probability * predicted_severity

    expense_load = 0.30
    profit_margin = 0.08

    suggested_premium = expected_loss / (1 - expense_load - profit_margin)
    loss_ratio = expected_loss / input_data["current_premium"]

    if loss_ratio < 0.60:
        risk_level = "Low Risk"
        action = "Rate appears adequate"
    elif loss_ratio < 0.85:
        risk_level = "Moderate Risk"
        action = "Review pricing"
    else:
        risk_level = "High Risk"
        action = "Consider rate increase or underwriting review"

    return {
        "claim_probability": claim_probability,
        "predicted_severity": predicted_severity,
        "expected_loss": expected_loss,
        "suggested_premium": suggested_premium,
        "loss_ratio": loss_ratio,
        "risk_level": risk_level,
        "action": action
    }


def scenario_inputs(prefix, defaults):
    driver_age = st.number_input(
        f"{prefix} Driver Age", min_value=18, max_value=90, value=defaults["driver_age"]
    )
    vehicle_age = st.number_input(
        f"{prefix} Vehicle Age", min_value=0, max_value=30, value=defaults["vehicle_age"]
    )
    annual_mileage = st.number_input(
        f"{prefix} Annual Mileage", min_value=1000, max_value=50000, value=defaults["annual_mileage"]
    )

    location = st.selectbox(
        f"{prefix} Location",
        ["Urban", "Suburban", "Rural"],
        index=["Urban", "Suburban", "Rural"].index(defaults["location"])
    )

    vehicle_type = st.selectbox(
        f"{prefix} Vehicle Type",
        ["Sedan", "SUV", "Truck", "Sports"],
        index=["Sedan", "SUV", "Truck", "Sports"].index(defaults["vehicle_type"])
    )

    coverage_type = st.selectbox(
        f"{prefix} Coverage Type",
        ["Basic", "Standard", "Premium"],
        index=["Basic", "Standard", "Premium"].index(defaults["coverage_type"])
    )

    prior_claims = st.number_input(
        f"{prefix} Prior Claims", min_value=0, max_value=10, value=defaults["prior_claims"]
    )
    traffic_violations = st.number_input(
        f"{prefix} Traffic Violations", min_value=0, max_value=10, value=defaults["traffic_violations"]
    )
    policy_tenure = st.number_input(
        f"{prefix} Policy Tenure", min_value=0, max_value=30, value=defaults["policy_tenure"]
    )
    current_premium = st.number_input(
        f"{prefix} Current Premium", min_value=300, max_value=10000, value=defaults["current_premium"]
    )

    return {
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "annual_mileage": annual_mileage,
        "location": location,
        "vehicle_type": vehicle_type,
        "coverage_type": coverage_type,
        "prior_claims": prior_claims,
        "traffic_violations": traffic_violations,
        "policy_tenure": policy_tenure,
        "current_premium": current_premium
    }


st.title("P&C Insurance Pricing AI")
st.write(
    "Compare two policy scenarios and estimate claim probability, severity, expected loss, "
    "suggested premium, and rate indication."
)

default_current = {
    "driver_age": 30,
    "vehicle_age": 5,
    "annual_mileage": 15000,
    "location": "Urban",
    "vehicle_type": "Sedan",
    "coverage_type": "Standard",
    "prior_claims": 0,
    "traffic_violations": 0,
    "policy_tenure": 3,
    "current_premium": 1200
}

default_proposed = {
    "driver_age": 30,
    "vehicle_age": 5,
    "annual_mileage": 18000,
    "location": "Urban",
    "vehicle_type": "SUV",
    "coverage_type": "Standard",
    "prior_claims": 1,
    "traffic_violations": 0,
    "policy_tenure": 3,
    "current_premium": 1200
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Scenario")
    current_input = scenario_inputs("Current", default_current)

with col2:
    st.subheader("Proposed Scenario")
    proposed_input = scenario_inputs("Proposed", default_proposed)

if st.button("Compare Scenarios"):

    current_result = calculate_pricing(current_input)
    proposed_result = calculate_pricing(proposed_input)

    st.subheader("Scenario Comparison")

    comparison_df = pd.DataFrame({
        "Metric": [
            "Claim Probability",
            "Predicted Severity",
            "Expected Loss",
            "Suggested Premium",
            "Loss Ratio",
            "Risk Level"
        ],
        "Current Scenario": [
            f"{current_result['claim_probability']:.2%}",
            f"${current_result['predicted_severity']:,.2f}",
            f"${current_result['expected_loss']:,.2f}",
            f"${current_result['suggested_premium']:,.2f}",
            f"{current_result['loss_ratio']:.2%}",
            current_result["risk_level"]
        ],
        "Proposed Scenario": [
            f"{proposed_result['claim_probability']:.2%}",
            f"${proposed_result['predicted_severity']:,.2f}",
            f"${proposed_result['expected_loss']:,.2f}",
            f"${proposed_result['suggested_premium']:,.2f}",
            f"{proposed_result['loss_ratio']:.2%}",
            proposed_result["risk_level"]
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    premium_change = proposed_result["suggested_premium"] - current_result["suggested_premium"]
    premium_change_pct = premium_change / current_result["suggested_premium"]

    expected_loss_change = proposed_result["expected_loss"] - current_result["expected_loss"]
    expected_loss_change_pct = expected_loss_change / current_result["expected_loss"]

    probability_change = proposed_result["claim_probability"] - current_result["claim_probability"]

    st.subheader("Rate Indication Summary")

    st.write(f"**Change in Claim Probability:** {probability_change:.2%}")
    st.write(f"**Change in Expected Loss:** ${expected_loss_change:,.2f} ({expected_loss_change_pct:.2%})")
    st.write(f"**Suggested Premium Change:** ${premium_change:,.2f} ({premium_change_pct:.2%})")

    if premium_change_pct > 0.10:
        st.error("Rate indication suggests a material increase.")
    elif premium_change_pct < -0.10:
        st.success("Rate indication suggests a material decrease.")
    else:
        st.warning("Rate indication suggests a moderate or stable change.")

    st.subheader("Top Risk Drivers")

    top_drivers = feature_importance.sort_values(
        by="importance",
        ascending=False
    ).head(10)

    st.dataframe(top_drivers, use_container_width=True)
    st.bar_chart(top_drivers.set_index("feature")["importance"])

    st.subheader("Top Protective Factors")

    protective_factors = feature_importance.sort_values(
        by="importance",
        ascending=True
    ).head(10)

    st.dataframe(protective_factors, use_container_width=True)
    st.bar_chart(protective_factors.set_index("feature")["importance"])

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit | Frequency × Severity Pricing Model")