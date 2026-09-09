import pandas as pd

df = pd.read_csv("data/amazon_intent_sample.csv")

df = df.sample(200, random_state=42).reset_index(drop=True)

df.insert(0, "id", range(1, len(df) + 1))

labels = {
    "delivery": "delivery_tracking",
    "arrived": "late_or_missing_delivery",
    "package": "delivery_tracking",
    "parcel": "delivery_tracking",
    "refund": "return_or_refund",
    "return": "return_or_refund",
    "charged": "payment_or_charge",
    "payment": "payment_or_charge",
    "prime": "prime_or_subscription",
    "cancel": "cancel_order",
    "cancelled": "cancel_order",
    "login": "account_or_security",
    "password": "account_or_security",
    "account": "account_or_security",
    "kindle": "product_problem",
    "damaged": "product_problem",
    "broken": "product_problem",
    "error": "technical_problem",
    "not working": "technical_problem"
}

def classify(text):
    text_lower = text.lower()

    for word, intent in labels.items():
        if word in text_lower:
            return intent

    return "other"

df["intent"] = df["text"].apply(classify)

df["escalate"] = df["text"].apply(
    lambda x: "Yes"
    if any(word in x.lower() for word in [
        "no response",
        "still waiting",
        "again",
        "20 times",
        "fraud",
        "complaint",
        "not resolved",
        "not resolved",
        "nobody",
        "no one"
    ])
    else "No"
)

df["escalation_reason"] = df.apply(
    lambda row:
    "Customer reports repeated or unresolved support"
    if row["escalate"] == "Yes"
    else "",
    axis=1
)

df.to_csv("data/golden_set.csv", index=False)

print("Created:", len(df), "examples")