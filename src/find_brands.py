import pandas as pd

file_path = "data/twcs.csv"

df = pd.read_csv(
    file_path,
    usecols=["author_id", "inbound"]
)

print("Total tweets:", len(df))

print("\nTop authors:")
print(df["author_id"].value_counts().head(30))

print("\nTweets by inbound/outbound:")
print(df["inbound"].value_counts())