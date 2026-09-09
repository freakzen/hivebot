import pandas as pd

df = pd.read_csv("data/twcs.csv")

tweet_id = 1

while tweet_id:
    row = df[df["tweet_id"] == tweet_id]

    if row.empty:
        break

    row = row.iloc[0]

    print("\nTweet ID:", row["tweet_id"])
    print("Author:", row["author_id"])
    print("Inbound:", row["inbound"])
    print("Text:", row["text"])
    print("Response:", row["response_tweet_id"])
    print("In response to:", row["in_response_to_tweet_id"])

    next_id = row["response_tweet_id"]

    if pd.isna(next_id):
        break

    tweet_id = int(next_id)