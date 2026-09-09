import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id"
    ]
)

tweets = df.set_index("tweet_id")

amazon = df[
    (df["author_id"] == "AmazonHelp") &
    (df["response_tweet_id"].notna())
]

responses = []

for _, row in amazon.iterrows():

    for response_id in str(row["response_tweet_id"]).split(","):

        try:
            response_id = int(response_id)
        except:
            continue

        if response_id not in tweets.index:
            continue

        customer = tweets.loc[response_id]

        if customer["inbound"] == True:
            responses.append({
                "customer": customer["text"],
                "amazon": row["text"]
            })

pairs = pd.DataFrame(responses)

print("Total pairs:", len(pairs))

print("\nSample historical resolutions:\n")

for _, row in pairs.sample(30, random_state=42).iterrows():
    print("CUSTOMER:")
    print(row["customer"])
    print("\nAMAZON:")
    print(row["amazon"])
    print("\n" + "=" * 80)