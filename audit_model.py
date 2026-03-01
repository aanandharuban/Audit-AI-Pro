import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# This ensures we can see all columns when we print data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Libraries loaded successfully!")