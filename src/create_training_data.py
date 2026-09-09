import pandas as pd

df = pd.read_csv(
    "data/twcs.csv",
    usecols=["author_id", "inbound", "text"]
)

customers = df[
    (df["inbound"] == True) &
    (df["text"].str.contains("@AmazonHelp", na=False))
].copy()

customers = customers[customers["text"].str.len() >= 10]

def assign_intent(text):
    text = text.lower()

    if any(x in text for x in [
        "cancel my order",
        "cancel order",
        "cancelled my order",
        "cancelling my order"
    ]):
        return "cancel_order"

    if any(x in text for x in [
        "refund",
        "money back",
        "return this",
        "return my",
        "send back",
        "returning"
    ]):
        return "return_or_refund"

    if any(x in text for x in [
        "charged",
        "charge",
        "payment",
        "credit card",
        "debit card",
        "billing"
    ]):
        return "payment_or_charge"

    if any(x in text for x in [
        "prime membership",
        "prime subscription",
        "prime account",
        "prime trial",
        "cancel prime"
    ]):
        return "prime_or_subscription"

    if any(x in text for x in [
        "password",
        "login",
        "log in",
        "sign in",
        "account hacked",
        "account security"
    ]):
        return "account_or_security"

    if any(x in text for x in [
        "app crash",
        "app isn't working",
        "app not working",
        "website isn't working",
        "website not working",
        "error message",
        "broken link"
    ]):
        return "technical_problem"

    if any(x in text for x in [
        "broken",
        "damaged",
        "defective",
        "defect",
        "doesn't work",
        "not working"
    ]):
        return "product_problem"

    if any(x in text for x in [
        "tracking number",
        "track my order",
        "track my package",
        "tracking information",
        "tracking link",
        "where is my package"
    ]):
        return "delivery_tracking"

    if any(x in text for x in [
        "hasn't arrived",
        "has not arrived",
        "not arrived",
        "haven't received",
        "have not received",
        "not received",
        "missing package",
        "package is late",
        "delivery is late",
        "late delivery"
    ]):
        return "late_or_missing_delivery"

    if any(x in text for x in [
        "wrong item",
        "wrong product",
        "ordered the wrong",
        "order is wrong",
        "order problem",
        "problem with my order"
    ]):
        return "order_problem"

    if any(x in text for x in [
        "no response",
        "nobody helped",
        "no one helped",
        "already contacted",
        "contacted you before",
        "multiple times",
        "still not resolved",
        "not resolved",
        "manager",
        "supervisor",
        "escalate",
        "complaint",
        "very disappointed",
        "terrible service"
    ]):
        return "complaint_or_unresolved"

    return "other"


customers["intent"] = customers["text"].apply(assign_intent)

golden = pd.read_csv("data/golden_set.csv")

golden_texts = set(golden["text"].str.lower())

customers = customers[
    ~customers["text"].str.lower().isin(golden_texts)
]

customers = customers.sample(
    min(50000, len(customers)),
    random_state=42
)

customers.to_csv(
    "data/training_data.csv",
    index=False
)

print("Training examples:", len(customers))
print("\nIntent distribution:")
print(customers["intent"].value_counts())