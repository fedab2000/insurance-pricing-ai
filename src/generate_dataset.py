import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 5000

# -----------------------------
# Generate customer / policy data
# -----------------------------
driver_age = np.random.randint(18, 80, n)
vehicle_age = np.random.randint(0, 20, n)
annual_mileage = np.random.normal(15000, 5000, n).clip(3000, 40000)

locations = np.random.choice(
    ["Urban", "Suburban", "Rural"],
    size=n,
    p=[0.45, 0.35, 0.20]
)

vehicle_type = np.random.choice(
    ["Sedan", "SUV", "Truck", "Sports"],
    size=n,
    p=[0.40, 0.35, 0.20, 0.05]
)

coverage_type = np.random.choice(
    ["Basic", "Standard", "Premium"],
    size=n,
    p=[0.30, 0.50, 0.20]
)

prior_claims = np.random.poisson(0.4, n)
traffic_violations = np.random.poisson(0.3, n)
policy_tenure = np.random.randint(0, 15, n)

# -----------------------------
# Claim probability logic
# -----------------------------
risk_score = (
    -3.0
    + 0.025 * (annual_mileage / 1000)
    + 0.08 * prior_claims
    + 0.12 * traffic_violations
    + 0.03 * vehicle_age
    - 0.015 * policy_tenure
)

risk_score += np.where(driver_age < 25, 0.8, 0)
risk_score += np.where(driver_age > 65, 0.3, 0)
risk_score += np.where(locations == "Urban", 0.5, 0)
risk_score += np.where(locations == "Rural", -0.2, 0)
risk_score += np.where(vehicle_type == "Sports", 0.8, 0)
risk_score += np.where(vehicle_type == "Truck", 0.3, 0)
risk_score += np.where(coverage_type == "Premium", 0.2, 0)

claim_probability = 1 / (1 + np.exp(-risk_score))
claim_flag = np.random.binomial(1, claim_probability)

# -----------------------------
# Claim severity logic
# -----------------------------
base_severity = np.random.gamma(shape=2.0, scale=2500, size=n)

severity_multiplier = (
    1
    + np.where(vehicle_type == "Sports", 0.8, 0)
    + np.where(vehicle_type == "Truck", 0.4, 0)
    + np.where(coverage_type == "Premium", 0.5, 0)
    + np.where(locations == "Urban", 0.2, 0)
)

claim_amount = claim_flag * base_severity * severity_multiplier
claim_amount = np.round(claim_amount, 2)

# -----------------------------
# Current premium logic
# -----------------------------
current_premium = (
    700
    + 20 * (annual_mileage / 1000)
    + 30 * prior_claims
    + 25 * traffic_violations
    + 10 * vehicle_age
)

current_premium += np.where(driver_age < 25, 500, 0)
current_premium += np.where(driver_age > 65, 150, 0)
current_premium += np.where(locations == "Urban", 300, 0)
current_premium += np.where(locations == "Rural", -100, 0)
current_premium += np.where(vehicle_type == "Sports", 700, 0)
current_premium += np.where(vehicle_type == "Truck", 250, 0)
current_premium += np.where(coverage_type == "Premium", 600, 0)
current_premium += np.where(coverage_type == "Basic", -200, 0)

current_premium = np.round(current_premium + np.random.normal(0, 100, n), 2)
current_premium = np.maximum(current_premium, 300)

# -----------------------------
# Final dataset
# -----------------------------
df = pd.DataFrame({
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "annual_mileage": np.round(annual_mileage, 0),
    "location": locations,
    "vehicle_type": vehicle_type,
    "coverage_type": coverage_type,
    "prior_claims": prior_claims,
    "traffic_violations": traffic_violations,
    "policy_tenure": policy_tenure,
    "current_premium": current_premium,
    "claim_flag": claim_flag,
    "claim_amount": claim_amount
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/insurance_pricing_dataset.csv", index=False)

print("Dataset created: data/insurance_pricing_dataset.csv")
print(df.head())
print("\nClaim rate:", round(df["claim_flag"].mean(), 3))
print("Average claim amount:", round(df[df["claim_amount"] > 0]["claim_amount"].mean(), 2))