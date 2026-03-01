import pandas as pd
import numpy as np

# Load the NEW larger dataset
df = pd.read_csv("indian_audit_data_v2.csv")

print("--- Engineering Advanced Features ---")

# 1. Basic Features (Same as before)
df['is_round_amount'] = df['Total'].apply(lambda x: 1 if x % 1000 == 0 else 0)
df['Date'] = pd.to_datetime(df['Date'])
df['is_weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# 2. NEW FEATURE: "Same Vendor, Same Day" Count
# This counts: "How many bills did 'Ramesh Traders' send us on 'Jan 12'?"
# If this number is > 1, it might be splitting bills!
df['vendor_daily_count'] = df.groupby(['Date', 'Party_Name'])['Total'].transform('count')

# 3. NEW FEATURE: "Total Daily Payment to Vendor"
# This sums up: "We paid Ramesh Traders a TOTAL of ₹50,000 today."
df['vendor_daily_total'] = df.groupby(['Date', 'Party_Name'])['Total'].transform('sum')

# Logic: If total is > 30,000 BUT individual bills are < 30,000, that is suspicious structuring!
df['is_structured_split'] = np.where(
    (df['vendor_daily_total'] > 30000) & (df['Total'] < 30000) & (df['vendor_daily_count'] > 1), 
    1, 0
)

# 4. Text Analysis
suspicious_words = ['cash', 'gift', 'personal', 'adjustment', 'split']
df['keyword_flag'] = df['Narration'].str.lower().apply(lambda x: 1 if any(w in x for w in suspicious_words) else 0)

# 5. Tax Math
df['expected_tax'] = df['Taxable_Value'] * (df['GST_Rate'] / 100)
df['actual_tax'] = df['CGST'] + df['SGST'] + df['IGST']
df['tax_mismatch'] = np.where(abs(df['expected_tax'] - df['actual_tax']) > 1.0, 1, 0)

# Select features for AI
features = df[['Taxable_Value', 'Total', 'is_round_amount', 'is_weekend', 'tax_mismatch', 'keyword_flag', 'vendor_daily_count', 'is_structured_split', 'Is_Fraud']]

features.to_csv("training_data_v2.csv", index=False)
print("Features Updated! Look for 'is_structured_split' in 'training_data_v2.csv'.")