import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker('en_IN')
rows = 5000
data = []

print("🚀 Generating Realistic Audit Dataset...")

for _ in range(rows):

    # 10% fraud base rate
    is_fraud = 1 if random.random() < 0.10 else 0
    date = fake.date_between(start_date='-2y', end_date='today')
    party = fake.company()

    # ==========================
    # AMOUNT GENERATION (OVERLAP)
    # ==========================
    amt = round(random.uniform(1000, 80000), 2)

    # 30% of ALL transactions (safe & fraud) can be round
    if random.random() < 0.30:
        amt = random.choice([5000, 10000, 15000, 25000, 50000])

    # ==========================
    # NARRATION GENERATION (OVERLAP)
    # ==========================
    safe_narrs = [
        "Monthly Office Rent",
        "Electricity Bill Payment",
        "Internet Charges",
        "Stationery Purchase",
        "Staff Salary",
        "Professional Charges"
    ]

    fraud_keywords = ['cash', 'personal', 'adjustment', 'urgent', 'split']

    if is_fraud:

        fraud_type = random.choice(['keyword', 'round', 'normal'])

        if fraud_type == 'keyword':
            narr = f"Payment for {random.choice(fraud_keywords)} expense"

        elif fraud_type == 'round':
            amt = random.choice([10000, 25000, 50000, 100000])
            narr = random.choice(safe_narrs)

        else:
            # Fraud that looks normal
            narr = f"Payment for {random.choice(safe_narrs)} - Inv #{random.randint(100,999)}"

    else:
        # SAFE transactions sometimes contain risky words too (5%)
        if random.random() < 0.05:
            narr = f"Petty cash reimbursement"
        else:
            narr = f"Payment for {random.choice(safe_narrs)} - Inv #{random.randint(100,999)}"

    # ==========================
    # FEATURE CALCULATION
    # ==========================
    is_round = 1 if amt % 1000 == 0 else 0
    is_q_end = 1 if date.month in [3, 6, 9, 12] else 0
    kw_flag = 1 if any(k in narr.lower() for k in fraud_keywords) else 0

    data.append([date, party, amt, narr, is_fraud, is_round, is_q_end, 1, kw_flag])

df = pd.DataFrame(data, columns=[
    'Date', 'Party', 'Total', 'Narration',
    'Is_Fraud', 'is_round', 'is_q_end', 'v_count', 'kw_flag'
])

df.to_csv("master_audit_dataset.csv", index=False)

print("✅ Created realistic 'master_audit_dataset.csv'")