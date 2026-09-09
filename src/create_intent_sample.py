import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=["author_id", "inbound", "text"]
)

customers = df[
    (df["inbound"] == True) &
    (df["text"].str.contains("@AmazonHelp", na=False))
]

sample = customers.sample(
    min(500, len(customers)),
    random_state=42
)

sample.to_csv(
    "data/amazon_intent_sample.csv",
    index=False
)

print("Saved:", len(sample), "customer messages")