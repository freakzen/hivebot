import pandas as pd

INPUT = "data/twcs.csv"
OUTPUT = "data/amazon_pairs.csv"

df = pd.read_csv(
    INPUT,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id"
    ]
)

df["text"] = df["text"].fillna("")

tweets = df.set_index("tweet_id")

amazon = df[
    (df["author_id"] == "AmazonHelp") &
    (df["response_tweet_id"].notna())
].copy()

pairs = []

for _, amazon_row in amazon.iterrows():

    response_ids = str(
        amazon_row["response_tweet_id"]
    ).split(",")

    for response_id in response_ids:

        try:
            response_id = int(response_id)
        except:
            continue

        if response_id not in tweets.index:
            continue

        customer = tweets.loc[response_id]

        if customer["inbound"] != True:
            continue

        if len(customer["text"]) < 10:
            continue

        pairs.append({
            "customer": customer["text"],
            "amazon": amazon_row["text"]
        })

pairs = pd.DataFrame(pairs)

pairs.to_csv(OUTPUT, index=False)

print("Created:", OUTPUT)
print("Rows:", len(pairs))
print("Size:", round(pairs.memory_usage(deep=True).sum() / 1024 / 1024, 2), "MB")