
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/cleaned_kc1.csv")

# Select 5 sample rows
sample = df.drop(columns=["defects"]).head(5)

# Save sample data
sample.to_csv("data/test_modules.csv", index=False)

print("Test CSV created successfully!")
