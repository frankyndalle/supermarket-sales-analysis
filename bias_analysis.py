import pandas as pd

df = pd.read_csv("supermarket_sales.csv.csv")

print("--- SUPERMARKET SALES BIAS & FAIRNESS REPORT ---")
print("\n[1] Gender Distribution per Branch (%):")
print((pd.crosstab(df["Branch"], df["Gender"], normalize="index") * 100).round(2))

print("\n[2] Payment Methods by Customer Type (%):")
print((pd.crosstab(df["Customer type"], df["Payment"], normalize="index") * 100).round(2))

print("\n[3] Revenue Contribution by Product Line (%):")
revenue_prod = df.groupby("Product line")["Total"].sum()
print(((revenue_prod / revenue_prod.sum()) * 100).round(2))
