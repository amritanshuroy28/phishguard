import pandas as pd
df = pd.read_csv("data/Phishing_Legitimate_full.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Label column: {df.columns[-1]}")
print(df[df.columns[-1]].value_counts())
print(f"\nFirst 10 columns: {df.columns[:10].tolist()}")
