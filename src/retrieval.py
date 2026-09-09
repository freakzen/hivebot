import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = "data/amazon_pairs.csv"

pairs = pd.read_csv(DATA)

pairs["customer"] = pairs["customer"].fillna("")
pairs["amazon"] = pairs["amazon"].fillna("")

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