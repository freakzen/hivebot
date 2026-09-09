import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/golden_set.csv")

def classify(text):
    text = text.lower()

    if any(x in text for x in ["cancel", "cancelling", "cancelled"]):
        return "cancel_order"

    if any(x in text for x in ["refund", "return", "money back", "pick up"]):
        return "return_or_refund"

    if any(x in text for x in ["payment", "charged", "charge", "credit card", "paid"]):
        return "payment_or_charge"

    if any(x in text for x in ["prime", "subscription", "2-day shipping"]):
        return "prime_or_subscription"

    if any(x in text for x in ["login", "log in", "password", "account", "phone number"]):
        return "account_or_security"

    if any(x in text for x in ["error", "app", "website", "link", "can't", "cannot"]):
        return "technical_problem"

    if any(x in text for x in ["kindle", "fire", "broken", "damaged", "defect"]):
        return "product_problem"

    if any(x in text for x in ["tracking", "tracker", "carrier", "delivered", "delivery"]):
        return "delivery_tracking"

    if any(x in text for x in ["late", "haven't received", "not received", "not arrived", "waiting"]):
        return "late_or_missing_delivery"

    if any(x in text for x in ["order", "ordered", "preorder"]):
        return "order_problem"

    if any(x in text for x in ["complaint", "disappointed", "awful", "pathetic", "not helpful"]):
        return "complaint_or_unresolved"

    return "other"


df["prediction"] = df["text"].fillna("").apply(classify)

print("Accuracy:", accuracy_score(df["intent"], df["prediction"]))

print("\nClassification Report:\n")
print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)