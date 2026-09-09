import pandas as pd
import re
from retrieval import retrieve

df = pd.read_csv("data/golden_set.csv")

def normalize(text):
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

rows = []

for _, row in df.iterrows():

    query_normalized = normalize(row["text"])

    results = retrieve(row["text"], k=20)

    results = [
        r for r in results
        if normalize(r["customer"]) != query_normalized
    ]

    results = results[:1]

    if results:
        best = results[0]

        rows.append({
            "id": row["id"],
            "customer": row["text"],
            "historical_customer": best["customer"],
            "historical_response": best["amazon"],
            "similarity": best["similarity"]
        })

output = pd.DataFrame(rows)

output = output.sort_values(
    "similarity",
    ascending=False
)

output.to_csv(
    "data/retrieval_evaluation.csv",
    index=False
)

print("Saved:", len(output), "retrieval evaluations")

print("\nTop 10 matches:\n")

for _, row in output.head(10).iterrows():

    print("=" * 80)

    print("ID:", row["id"])
    print("Similarity:", round(row["similarity"], 3))

    print("\nCUSTOMER:")
    print(row["customer"])

    print("\nHISTORICAL CUSTOMER:")
    print(row["historical_customer"])

    print("\nHISTORICAL AMAZON RESPONSE:")
    print(row["historical_response"])