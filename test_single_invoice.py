import pandas as pd
import joblib
import numpy as np

# 1. Load your trained AI
model = joblib.load('audit_model_v1.pkl')

print("--- AI AUDITOR: LIVE CHECK ---")
print("Enter transaction details to check for fraud.\n")

# 2. Get Input from You
amount = float(input("Enter Total Bill Amount (e.g., 50000): "))
gst_rate = float(input("Enter GST Rate (e.g., 18): "))
narration = input("Enter Narration (e.g., 'Office Lunch'): ")
is_weekend_input = input("Is the date a Saturday/Sunday? (yes/no): ")

# 3. Convert your input into "Features" (The same math we did before)
# Logic: Check if round number
is_round = 1 if amount % 1000 == 0 else 0

# Logic: Check for weekend
is_weekend = 1 if is_weekend_input.lower() == 'yes' else 0

# Logic: Check for suspicious words
suspicious_words = ['cash', 'gift', 'personal', 'adjustment']
keyword_flag = 1 if any(w in narration.lower() for w in suspicious_words) else 0

# Logic: Check Tax Mismatch (We estimate taxable value backwards from total)
# Total = Taxable * (1 + Rate/100)  ->  Taxable = Total / (1 + Rate/100)
estimated_taxable = amount / (1 + (gst_rate/100))
expected_tax = estimated_taxable * (gst_rate/100)
# In a real manual entry, users might just type '5000' tax without calculating.
# For this demo, we assume the math is correct unless you force a mismatch in the next step.
tax_mismatch = 0 

# Let's create the data row for the AI
# Note: The order of columns MUST match the training data exactly!
# ['Taxable_Value', 'GST_Rate', 'Total', 'is_round_amount', 'is_weekend', 'tax_mismatch', 'keyword_flag']

input_data = pd.DataFrame([[
    estimated_taxable, 
    gst_rate, 
    amount, 
    is_round, 
    is_weekend, 
    tax_mismatch, 
    keyword_flag
]], columns=['Taxable_Value', 'GST_Rate', 'Total', 'is_round_amount', 'is_weekend', 'tax_mismatch', 'keyword_flag'])

# 4. Ask the AI
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

print("\n-----------------------------")
if prediction[0] == 1:
    print(f"🚨 ALERT: This transaction is SUSPICIOUS!")
    print(f"Confidence: {probability[0][1]*100:.2f}% sure it is fraud.")
    if keyword_flag: print("- Reason: Suspicious words found in narration.")
    if is_round: print("- Reason: Amount is a round number.")
else:
    print(f"✅ CLEAN: This transaction looks normal.")
    print(f"Confidence: {probability[0][0]*100:.2f}% safe.")
print("-----------------------------")