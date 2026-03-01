import pandas as pd
import random
from faker import Faker

fake = Faker('en_IN')

def create_demo_ledger(n=50):
    data = []
    # Standard Indian Expense Heads
    expenses = ['Professional Fees', 'Office Rent', 'Electricity Expense', 'Conveyance', 'Staff Welfare']
    
    for i in range(n):
        date = fake.date_between(start_date='-1m', end_date='today')
        party = fake.company()
        amount = round(random.uniform(2000, 45000), 2)
        narration = f"Being payment for {random.choice(expenses)}"
        data.append([date, party, amount, narration])
    
    # INJECT SPECIFIC FRAUDS FOR THE DEMO
    # 1. TDS Split (Two bills same day same party)
    d = fake.date_this_month()
    p = "Shree Balaji Traders"
    data.append([d, p, 28500.00, "Being professional charges part 1"])
    data.append([d, p, 29000.00, "Being professional charges part 2"])
    
    # 2. Personal Expense
    data.append([fake.date_this_month(), "Reliance Retail", 12500.00, "Personal grocery and gift items - Ramesh"])
    
    # 3. Round Number / Anomaly
    data.append([fake.date_this_month(), "Misc Vendor", 100000.00, "Adjustment entry"])

    df = pd.DataFrame(data, columns=['Voucher_Date', 'Particulars', 'Amount', 'Description'])
    df.to_csv("client_ledger_demo.csv", index=False)
    print("Demo Ledger Created: 'client_ledger_demo.csv'")

create_demo_ledger()