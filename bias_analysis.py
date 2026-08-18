import pandas as pd

# Load dataset
df = pd.read_csv("supermarket_sales.csv.csv")

print("=" * 60)
print("SUPERMARKET SALES: EXPLICIT BIAS & FAIRNESS DETECTION REPORT")
print("=" * 60)

# 1. Gender Representation across Branches (Detection)
print("\n[DETECTION 1] Gender Distribution per Branch (%):")
gender_branch = (pd.crosstab(df["Branch"], df["Gender"], normalize="index") * 100).round(2)
print(gender_branch)
print("-> Mitigation Strategy: Ensure balanced marketing campaigns and staff distribution to prevent regional gender skews across branches A, B, and C.")

# 2. Payment Method Preferences by Customer Type (Detection)
print("\n[DETECTION 2] Payment Methods by Customer Type (%):")
payment_cust = (pd.crosstab(df["Customer type"], df["Payment"], normalize="index") * 100).round(2)
print(payment_cust)
print("-> Mitigation Strategy: Implement targeted promotions across Ewallet and Cash payment channels to reduce reliance on single payment gateways for specific customer tiers.")

# 3. Revenue Contribution by Product Line (Detection & Disparity)
print("\n[DETECTION 3] Revenue Contribution by Product Line (%):")
revenue_prod = df.groupby("Product line")["Total"].sum()
revenue_share = ((revenue_prod / revenue_prod.sum()) * 100).round(2)
print(revenue_share)
print("-> Mitigation Strategy: Address revenue imbalances by balancing inventory allocation, avoiding over-indexing on underperforming categories, and standardizing promotional support.")

print("\n" + "=" * 60)
print("Bias Detection and Mitigation Analysis Complete.")
print("=" * 60)
