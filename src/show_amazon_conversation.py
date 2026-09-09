import pandas as pd

df = pd.read_csv("data/twcs.csv")

tweets = df.set_index("tweet_id")

amazon = df[
    (df["author_id"] == "AmazonHelp") &
    (df["response_tweet_id"].notna())
]

best = None
best_length = 0

for _, start in amazon.sample(500, random_state=42).iterrows():

    conversation = []
    tweet_id = int(start["tweet_id"])

    for _ in range(20):

        if tweet_id not in tweets.index:
            break

        row = tweets.loc[tweet_id]

        conversation.append(row)

        next_id = row["response_tweet_id"]

        if pd.isna(next_id):
            break

        next_id = str(next_id).split(",")[0]

        tweet_id = int(next_id)

    if len(conversation) > best_length:
        best = conversation
        best_length = len(conversation)

print("Conversation length:", best_length)

for row in best:

    speaker = "CUSTOMER" if row["inbound"] else "AMAZON"

    print(f"\n{speaker}:")
    print(row["text"])