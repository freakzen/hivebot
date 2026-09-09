import pandas as pd
import re

df = pd.read_csv("data/retrieval_evaluation.csv")


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_keywords(text):
    words = normalize(text).split()

    stopwords = {
        "the", "a", "an", "is", "it", "i", "im", "my",
        "to", "and", "of", "for", "on", "in", "this",
        "that", "with", "please", "amazon", "help"
    }

    return set(w for w in words if len(w) > 2 and w not in stopwords)


results = []

for _, row in df.iterrows():

    customer_words = get_keywords(row["customer"])
    historical_words = get_keywords(row["historical_customer"])

    if not customer_words:
        relevance = 1
    else:
        overlap = len(customer_words & historical_words) / len(customer_words)

        if overlap >= 0.6:
            relevance = 5
        elif overlap >= 0.4:
            relevance = 4
        elif overlap >= 0.25:
            relevance = 3
        elif overlap >= 0.1:
            relevance = 2
        else:
            relevance = 1

    response = normalize(row["historical_response"])

    useful_signals = [
        "please",
        "contact",
        "check",
        "provide",
        "reply",
        "reach",
        "help",
        "order",
        "tracking",
        "refund",
        "return",
        "account",
        "delivery"
    ]

    signal_count = sum(
        1 for signal in useful_signals
        if signal in response
    )

    if signal_count >= 4:
        helpfulness = 5
    elif signal_count >= 3:
        helpfulness = 4
    elif signal_count >= 2:
        helpfulness = 3
    elif signal_count >= 1:
        helpfulness = 2
    else:
        helpfulness = 1

    grounding = 5 if row["similarity"] >= 0.5 else (
        4 if row["similarity"] >= 0.4 else (
        3 if row["similarity"] >= 0.3 else (
        2 if row["similarity"] >= 0.2 else 1)))

    appropriateness = 4 if helpfulness >= 3 else 2

    results.append({
        "id": row["id"],
        "customer": row["customer"],
        "historical_customer": row["historical_customer"],
        "historical_response": row["historical_response"],
        "similarity": row["similarity"],
        "relevance": relevance,
        "grounding": grounding,
        "helpfulness": helpfulness,
        "appropriateness": appropriateness
    })


output = pd.DataFrame(results)

output.to_csv(
    "data/local_judge_results.csv",
    index=False
)

print("\n=== LOCAL JUDGE RESULTS ===")

print("Examples evaluated:", len(output))

for metric in [
    "relevance",
    "grounding",
    "helpfulness",
    "appropriateness"
]:
    print(
        metric.capitalize() + ":",
        round(output[metric].mean(), 2),
        "/ 5"
    )

print("\nSaved:")
print("data/local_judge_results.csv")