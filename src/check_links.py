import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

amazon = df[df["author_id"] == "AmazonHelp"]

print("Amazon tweets:", len(amazon))

print(
    "\nAmazon tweets with response_tweet_id:",
    amazon["response_tweet_id"].notna().sum()
)

print(
    "Amazon tweets with in_response_to_tweet_id:",
    amazon["in_response_to_tweet_id"].notna().sum()
)

print("\nExample Amazon → response links:\n")

sample = amazon[amazon["response_tweet_id"].notna()].head(10)

for _, row in sample.iterrows():
    print("Amazon ID:", row["tweet_id"])
    print("Amazon:", row["text"])
    print("response_tweet_id:", row["response_tweet_id"])

    ids = str(row["response_tweet_id"]).split(",")

    for tweet_id in ids[:3]:
        try:
            response = df[df["tweet_id"] == int(tweet_id)]

            if not response.empty:
                r = response.iloc[0]
                print("→", r["author_id"], ":", r["text"])

        except:
            pass

    print("-" * 80)