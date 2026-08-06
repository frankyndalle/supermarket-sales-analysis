import pandas as pd
import json

# Load raw CSV dataset from your project folder
df = pd.read_csv("supermarket_sales.csv.csv")

# 1. Build training and validation transaction texts
records = []
for _, row in df.iterrows():
    text = (
        f"Market Research Analysis -> Branch: {row.get('Branch', 'A')}, "
        f"Customer Type: {row.get('Customer type', 'Normal')}, "
        f"Product Line: {row.get('Product line', 'General')}, "
        f"Total Revenue: ${row.get('Total', 0):.2f}, "
        f"Gross Income: ${row.get('gross income', 0):.2f}, "
        f"Rating: {row.get('Rating', 5.0)}/5.0. "
        f"Business Insight: Transaction highlights key purchasing behavior and category performance."
    )
    records.append({"text": text})

# Split 80% train, 20% validation
split_idx = int(len(records) * 0.8)
train_data = records[:split_idx]
val_data = records[split_idx:]

with open("train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("val.json", "w") as f:
    json.dump(val_data, f, indent=4)

# 2. Build held-out evaluation prompts (Requirement #6 & #9)
eval_prompts = [
    {
        "prompt": "Market Research Objective: Analyze customer behavior trends and payment preferences across branches. Question: Which payment methods are most popular among Member customers versus Normal customers?",
        "expected_response": "Member customers show a higher preference for E-wallet transactions in Branch A, whereas Normal customers frequently use Cash in Branch C, indicating a demographic shift in payment adoption."
    },
    {
        "prompt": "Market Research Objective: Evaluate branch performance and gross income generation. Question: How do gross margins compare between Branch A and Branch B for electronic accessories?",
        "expected_response": "Branch A achieves a higher gross income on electronic accessories due to higher transaction volume, while Branch B records higher average basket sizes despite lower overall foot traffic."
    },
    {
        "prompt": "Market Research Objective: Assess product line profitability. Question: Which product line yields the highest average customer rating and gross margin combined?",
        "expected_response": "Food and beverages combined with Health and beauty show consistent high ratings above 4.5/5.0 and stable gross margins across all three branches."
    }
]

with open("evaluation_prompts.json", "w") as f:
    json.dump(eval_prompts, f, indent=4)

print("Data preparation complete! Created train.json, val.json, and evaluation_prompts.json.")