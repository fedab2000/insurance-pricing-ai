import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="P&C Pricing Dashboard", layout="wide")

# Load models and data
frequency_model = joblib.load("outputs/frequency_model.joblib")
severity_model = joblib.load("outputs/severity_model.joblib")
feature_importance = pd.read_csv("outputs/frequency_feature_importance.csv")
pricing_data = pd.read_csv("data/insurance_pricing_dataset.csv")

st.title("P&C Pricing Analytics Dashboard")

st.write(
    "This dashboard supports pricing analysis using frequency, severity, expected loss, "
    "risk drivers, and rate indication analytics."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pricing Calculator",
    "Scenario Comparison",
    "Risk Drivers",
    "Rate Indication",
    "Model Info"
])


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


def get_policy_inputs(prefix=""):
    driver_age = st.number_input(f"{prefix}Driver Age", 18, 90, 30)
    vehicle_age = st.number_input(f"{prefix}Vehicle Age", 0, 30, 5)
    annual_mileage = st.number_input(f"{prefix}Annual Mileage", 1000, 50000, 15000)

    location = st.selectbox(f"{prefix}Location", ["Urban", "Suburban", "Rural"])
    vehicle_type = st.selectbox(f"{prefix}Vehicle Type", ["Sedan", "SUV", "Truck", "Sports"])
    coverage_type = st.selectbox(f"{prefix}Coverage Type", ["Basic", "Standard", "Premium"])

    prior_claims = st.number_input(f"{prefix}Prior Claims", 0, 10, 0)
    traffic_violations = st.number_input(f"{prefix}Traffic Violations", 0, 10, 0)
    policy_tenure = st.number_input(f"{prefix}Policy Tenure", 0, 30, 3)
    current_premium = st.number_input(f"{prefix}Current Premium", 300, 10000, 1200)

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


with tab1:
    st.header("Pricing Calculator")

    input_data = get_policy_inputs()

    if st.button("Calculate Premium"):
        result = calculate_pricing(input_data)

        st.subheader("Pricing Results")
        st.write(f"**Claim Probability:** {result['claim_probability']:.2%}")
        st.write(f"**Predicted Severity:** ${result['predicted_severity']:,.2f}")
        st.write(f"**Expected Loss:** ${result['expected_loss']:,.2f}")
        st.write(f"**Suggested Premium:** ${result['suggested_premium']:,.2f}")
        st.write(f"**Loss Ratio:** {result['loss_ratio']:.2%}")
        st.write(f"**Risk Level:** {result['risk_level']}")

        if result["risk_level"] == "Low Risk":
            st.success(result["action"])
        elif result["risk_level"] == "Moderate Risk":
            st.warning(result["action"])
        else:
            st.error(result["action"])


with tab2:
    st.header("Scenario Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Current Scenario")
        current_input = get_policy_inputs("Current ")

    with col2:
        st.subheader("Proposed Scenario")
        proposed_input = get_policy_inputs("Proposed ")

    if st.button("Compare Scenarios"):
        current_result = calculate_pricing(current_input)
        proposed_result = calculate_pricing(proposed_input)

        comparison_df = pd.DataFrame({
            "Metric": [
                "Claim Probability",
                "Predicted Severity",
                "Expected Loss",
                "Suggested Premium",
                "Loss Ratio",
                "Risk Level"
            ],
            "Current": [
                f"{current_result['claim_probability']:.2%}",
                f"${current_result['predicted_severity']:,.2f}",
                f"${current_result['expected_loss']:,.2f}",
                f"${current_result['suggested_premium']:,.2f}",
                f"{current_result['loss_ratio']:.2%}",
                current_result["risk_level"]
            ],
            "Proposed": [
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

        st.subheader("Rate Indication Summary")
        st.write(f"**Suggested Premium Change:** ${premium_change:,.2f}")
        st.write(f"**Suggested Premium Change %:** {premium_change_pct:.2%}")


with tab3:
    st.header("Risk Drivers")

    top_drivers = feature_importance.sort_values(
        by="importance",
        ascending=False
    ).head(10)

    protective_factors = feature_importance.sort_values(
        by="importance",
        ascending=True
    ).head(10)

    st.subheader("Top Risk Drivers")
    st.dataframe(top_drivers, use_container_width=True)
    st.bar_chart(top_drivers.set_index("feature")["importance"])

    st.subheader("Top Protective Factors")
    st.dataframe(protective_factors, use_container_width=True)
    st.bar_chart(protective_factors.set_index("feature")["importance"])


with tab4:
    st.header("Rate Indication Analysis")

    target_loss_ratio = st.slider(
        "Target Loss Ratio",
        min_value=0.40,
        max_value=0.90,
        value=0.65,
        step=0.01
    )

    segment_col = st.selectbox(
        "Select Segment",
        ["location", "vehicle_type", "coverage_type"]
    )

    segment = pricing_data.groupby(segment_col).agg(
        policies=("claim_flag", "count"),
        total_premium=("current_premium", "sum"),
        total_claims=("claim_amount", "sum"),
        claim_count=("claim_flag", "sum")
    ).reset_index()

    segment["loss_ratio"] = segment["total_claims"] / segment["total_premium"]
    segment["claim_frequency"] = segment["claim_count"] / segment["policies"]
    segment["avg_premium"] = segment["total_premium"] / segment["policies"]
    segment["indicated_rate_change"] = (
        segment["loss_ratio"] / target_loss_ratio - 1
    )

    st.dataframe(segment, use_container_width=True)

    st.subheader("Indicated Rate Change by Segment")
    st.bar_chart(segment.set_index(segment_col)["indicated_rate_change"])


with tab5:
    st.header("Model Info")

    st.write("### Frequency Model")
    st.write("Predicts the probability that a policy will have a claim.")
    st.write("Model used: Logistic Regression with TF-style preprocessing for tabular features.")

    st.write("### Severity Model")
    st.write("Predicts the expected claim amount for policies with claims.")
    st.write("Model used: Gradient Boosting Regressor.")

    st.write("### Pricing Formula")
    st.code(
        "Expected Loss = Claim Probability × Predicted Severity\n"
        "Suggested Premium = Expected Loss / (1 - Expense Load - Profit Margin)"
    )

    st.write("### Business Use Cases")
    st.write("- Pricing review")
    st.write("- Underwriting appetite analysis")
    st.write("- Rate indication analysis")
    st.write("- Scenario testing")
    st.write("- Risk driver interpretation")

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit | P&C Pricing Analytics Dashboard")