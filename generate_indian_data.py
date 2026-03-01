import pandas as pd
import random
from faker import Faker
import datetime

fake = Faker('en_IN')

# We are going BIG now. 5000 transactions.
num_transactions = 5000
data = []
gst_rates = [0, 5, 12, 18, 28]

# Suspicious words
suspicious_narrations = ["Personal", "Cash", "Gift", "Adjustment", "Ref"]

print(f"Generating {num_transactions} transactions with COMPLEX frauds...")

# We need a list of vendors to reuse (so we can have repeat payments)
vendors = [fake.company() for _ in range(50)]

for _ in range(num_transactions):
    date = fake.date_between(start_date='-1y', end_date='today')
    party_name = random.choice(vendors) # Pick an existing vendor
    gstin = f"{random.randint(10,38)}{fake.bothify(text='?????####?')}{random.randint(1,9)}Z{random.choice(['A','B','C'])}"
    
    # FRAUD TYPE 4: THE "TDS DODGE" (Split Transactions)
    # 2% chance we create a "Split Bill" fraud
    if random.random() < 0.02: 
        # The REAL amount was 50,000 (which needs TDS)
        # We split it into two bills of 25,000 to hide it.
        for i in range(2):
            taxable_value = 25000.00 
            rate = 18
            gst_amt = taxable_value * 0.18
            narration = "Split Payment Part " + str(i+1)
            
            # This is technically fraud because it hides the big amount
            is_fraud = 1 
            
            # Calculate Total
            total = taxable_value + gst_amt
            
            # Add to list
            data.append([date, party_name, gstin, taxable_value, rate, gst_amt/2, gst_amt/2, 0, total, narration, is_fraud])
            
        continue # Skip the rest of the loop

    # --- NORMAL RANDOM GENERATION (Existing Logic) ---
    is_fraud = random.random() < 0.03 # 3% other frauds
    taxable_value = round(random.uniform(1000, 40000), 2)
    rate = random.choice(gst_rates)
    gst_amt = taxable_value * (rate / 100)
    narration = f"Inv No {random.randint(1000,9999)}"

    if is_fraud:
        if random.random() < 0.5:
            narration = "Personal Expense - " + fake.first_name()
        else:
            taxable_value = 50000.00 # Round number fraud
            gst_amt = 9000.00

    # Tax Calculation
    if random.choice([True, False]): # Interstate
        cgst, sgst, igst = 0, 0, gst_amt
    else:
        cgst, sgst, igst = gst_amt/2, gst_amt/2, 0

    total = taxable_value + cgst + sgst + igst
    data.append([date, party_name, gstin, taxable_value, rate, cgst, sgst, igst, total, narration, 1 if is_fraud else 0])

# Save
columns = ['Date', 'Party_Name', 'GSTIN', 'Taxable_Value', 'GST_Rate', 'CGST', 'SGST', 'IGST', 'Total', 'Narration', 'Is_Fraud']
df = pd.DataFrame(data, columns=columns)
df.to_csv("indian_audit_data_v2.csv", index=False)
print("Done! Generated 'indian_audit_data_v2.csv' with Split Transaction Frauds.")