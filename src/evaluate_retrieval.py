import pandas as pd
from retrieval import retrieve

df = pd.read_csv("data/golden_set.csv")

scores = []

for _, row in df.iterrows():

    results = retrieve(row["text"], k=20)

    query = row["text"].strip().lower()

    results = [
        r for r in results
        if r["customer"].strip().lower() != query
    ]

    if results:
        scores.append(results[0]["similarity"])

print("\n=== RETRIEVAL EVALUATION ===")

print("Examples evaluated:", len(scores))

print("Average top-1 similarity:", round(sum(scores) / len(scores), 3))

print("Minimum similarity:", round(min(scores), 3))

print("Maximum similarity:", round(max(scores), 3))

for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:

    good = sum(score >= threshold for score in scores)

    print(
        f"Similarity >= {threshold}: "
        f"{good}/{len(scores)} "
        f"({round(good / len(scores) * 100, 1)}%)"
    )