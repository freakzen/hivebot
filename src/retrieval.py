import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

print("Historical customer-response pairs:", len(pairs))

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000
)

matrix = vectorizer.fit_transform(
    pairs["customer"]
)


def retrieve(query, k=5):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        matrix
    )[0]

    top_indices = scores.argsort()[-k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "customer": pairs.iloc[index]["customer"],
            "amazon": pairs.iloc[index]["amazon"],
            "similarity": scores[index]
        })

    return results


if __name__ == "__main__":

    query = input("Customer message: ")

    results = retrieve(query)

    print("\nHistorical examples:\n")

    for result in results:

        print(
            "Similarity:",
            round(result["similarity"], 3)
        )

        print("\nCUSTOMER:")
        print(result["customer"])

        print("\nAMAZON:")
        print(result["amazon"])

        print("-" * 80)