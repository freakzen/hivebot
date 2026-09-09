import pandas as pd

df = pd.read_csv("data/amazon_intent_sample.csv")

for i, row in df.head(50).iterrows():
    print(f"\n{i + 1}. {row['text']}")