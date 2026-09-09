import pandas as pd
import re

df = pd.read_csv(
    "data/twcs.csv",
    usecols=["author_id", "inbound", "text"]
)

customers = df[
    (df["inbound"] == True) &
    (df["text"].notna()) &
    (df["text"].str.contains("@AmazonHelp", na=False))
].copy()

golden = pd.read_csv("data/golden_set.csv")

golden_texts = set(
    golden["text"]
    .fillna("")
    .str.lower()
    .str.strip()
)

def clean(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def classify(text):

    t = clean(text)

    rules = {
        "cancel_order": [
            "cancel my order",
            "cancel order",
            "cancelled my order",
            "cancelling my order",
            "want to cancel",
            "need to cancel"
        ],

        "return_or_refund": [
            "refund",
            "money back",
            "return this",
            "return my",
            "send back",
            "return an item",
            "return the item"
        ],

        "payment_or_charge": [
            "charged twice",
            "charged me",
            "wrong charge",
            "unexpected charge",
            "payment failed",
            "payment declined",
            "credit card",
            "debit card"
        ],

        "account_or_security": [
            "account hacked",
            "can't login",
            "cannot login",
            "can't log in",
            "cannot log in",
            "forgot password",
            "reset password",
            "password",
            "account security"
        ],

        "prime_or_subscription": [
            "prime membership",
            "prime subscription",
            "prime trial",
            "cancel prime",
            "prime charged",
            "prime charge"
        ],

        "product_problem": [
            "product is broken",
            "product broken",
            "product damaged",
            "item damaged",
            "item is damaged",
            "defective product",
            "defective item",
            "product doesn't work",
            "product not working"
        ],

        "technical_problem": [
            "app is not working",
            "app not working",
            "app doesn't work",
            "website not working",
            "website doesn't work",
            "error message",
            "technical issue",
            "broken link",
            "login error"
        ],

        "delivery_tracking": [
            "track my order",
            "track my package",
            "tracking number",
            "tracking information",
            "tracking link",
            "where is my order",
            "where is my package",
            "tracking update"
        ],

        "late_or_missing_delivery": [
            "order hasn't arrived",
            "order has not arrived",
            "package hasn't arrived",
            "package has not arrived",
            "haven't received my order",
            "have not received my order",
            "haven't received my package",
            "have not received my package",
            "package is late",
            "delivery is late",
            "late delivery",
            "missing package"
        ],

        "order_problem": [
            "wrong item",
            "wrong product",
            "wrong order",
            "problem with my order",
            "issue with my order",
            "order is wrong"
        ],

        "complaint_or_unresolved": [
            "already contacted",
            "contacted before",
            "contacted multiple times",
            "contacted several times",
            "no response",
            "nobody helped",
            "no one helped",
            "still unresolved",
            "not resolved",
            "not helpful",
            "keep following up",
            "manager",
            "supervisor",
            "escalate",
            "terrible service",
            "awful service",
            "very disappointed"
        ]
    }

    matches = []

    for intent, keywords in rules.items():
        for keyword in keywords:
            if keyword in t:
                matches.append(intent)
                break

    if len(matches) == 1:
        return matches[0]

    return None


customers["intent"] = customers["text"].apply(classify)

customers = customers[
    customers["intent"].notna()
].copy()

customers["normalized"] = (
    customers["text"]
    .fillna("")
    .str.lower()
    .str.strip()
)

customers = customers[
    ~customers["normalized"].isin(golden_texts)
]

customers = customers.drop_duplicates(
    subset=["normalized"]
)

max_per_class = 3000

parts = []

for intent, group in customers.groupby("intent"):

    sample_size = min(len(group), max_per_class)

    parts.append(
        group.sample(
            sample_size,
            random_state=42
        )
    )

balanced = pd.concat(
    parts,
    ignore_index=True
)

balanced[
    ["text", "intent"]
].to_csv(
    "data/training_data_v2.csv",
    index=False
)

print("\n=== TRAINING DATA V2 ===")

print("Total examples:", len(balanced))

print("\nClass distribution:")

print(
    balanced["intent"]
    .value_counts()
    .to_string()
)

print("\nSaved:")
print("data/training_data_v2.csv")