import pandas as pd

df = pd.read_csv("data/amazon_intent_sample.csv")

df = df.head(50)

labels = [
    "product_problem",
    "technical_problem",
    "delivery_tracking",
    "other",
    "complaint_or_unresolved",
    "late_or_missing_delivery",
    "prime_or_subscription",
    "cancel_order",
    "delivery_tracking",
    "complaint_or_unresolved",
    "other",
    "other",
    "account_or_security",
    "other",
    "delivery_tracking",
    "complaint_or_unresolved",
    "return_or_refund",
    "delivery_tracking",
    "delivery_tracking",
    "other",
    "delivery_tracking",
    "return_or_refund",
    "order_problem",
    "complaint_or_unresolved",
    "other",
    "prime_or_subscription",
    "other",
    "cancel_order",
    "late_or_missing_delivery",
    "order_problem",
    "complaint_or_unresolved",
    "delivery_tracking",
    "complaint_or_unresolved",
    "delivery_tracking",
    "payment_or_charge",
    "complaint_or_unresolved",
    "delivery_tracking",
    "payment_or_charge",
    "complaint_or_unresolved",
    "other",
    "other",
    "complaint_or_unresolved",
    "return_or_refund",
    "prime_or_subscription",
    "late_or_missing_delivery",
    "other",
    "complaint_or_unresolved",
    "order_problem",
    "order_problem",
    "complaint_or_unresolved"
]

df.insert(0, "id", range(1, len(df) + 1))
df["intent"] = labels

df.to_csv("data/intent_labels.csv", index=False)

print("Created data/intent_labels.csv")