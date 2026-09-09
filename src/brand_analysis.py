import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "in_response_to_tweet_id"
    ]
)

brands = df.loc[~df["inbound"], "author_id"].value_counts().head(20).index

results = []

for brand in brands:
    brand_tweets = df[df["author_id"] == brand]
    brand_ids = set(brand_tweets["tweet_id"])

    customer_tweets = df[
        (df["inbound"]) &
        (df["in_response_to_tweet_id"].isin(brand_ids))
    ]

    results.append({
        "brand": brand,
        "brand_tweets": len(brand_tweets),
        "customer_replies": len(customer_tweets),
        "unique_customers": customer_tweets["author_id"].nunique()
    })

result = pd.DataFrame(results)

print(result.to_string(index=False))