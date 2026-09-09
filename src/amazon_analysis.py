import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id"
    ]
)

amazon_ids = set(
    df.loc[df["author_id"] == "AmazonHelp", "tweet_id"]
)

amazon = df[
    (df["author_id"] == "AmazonHelp") |
    (df["in_response_to_tweet_id"].isin(amazon_ids))
]

customers = amazon[amazon["inbound"] == True]

print("Amazon tweets:", len(amazon))
print("Customer messages:", len(customers))

print("\nSample customer messages:\n")

sample = customers.sample(
    min(20, len(customers)),
    random_state=42
)

for text in sample["text"]:
    print("-", text)