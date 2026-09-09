import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = "data/training_data_v2.csv"

df = pd.read_csv(DATA)
df["text"] = df["text"].fillna("")

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=80000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["text"])
y = df["intent"]

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X, y)


def contains(t, phrases):
    return any(p in t for p in phrases)


def classify_intent(text):
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)

    # 1. ACCOUNT / SECURITY
    if contains(t, [
        "hacked",
        "hack",
        "fraud on my account",
        "fraudulent account",
        "unauthorized access",
        "account compromised",
        "password",
        "can't login",
        "cannot login",
        "couldn't login",
        "could not login",
        "can't sign in",
        "cannot sign in",
        "couldn't sign in",
        "could not sign in",
        "unable to login",
        "unable to sign in",
        "login issue",
        "login problem"
    ]):
        return "account_or_security"

    # 2. CANCEL ORDER
    if contains(t, [
        "cancel my order",
        "cancel the order",
        "cancel order",
        "cancelled my order",
        "canceled my order",
        "cancel & give me my money back",
        "cancel and give me my money back"
    ]):
        return "cancel_order"

    # 3. PRIME / SUBSCRIPTION
    if contains(t, [
        "prime membership",
        "prime member",
        "prime subscription",
        "prime now credit",
        "prime now credits",
        "prime credit",
        "prime discount",
        "prime renewal",
        "cancel prime",
        "subscription",
        "membership"
    ]):
        return "prime_or_subscription"

    # 4. PAYMENT
    if contains(t, [
        "charged twice",
        "double charged",
        "charged for",
        "payment declined",
        "payment rejected",
        "card declined",
        "credit card",
        "debit card",
        "money deducted",
        "money taken",
        "amount deducted",
        "amount charged",
        "billing",
        "billing issue",
        "payment issue",
        "payment problem",
        "transfer the money",
        "transfer money to my account",
        "money in my account",
        "rs",
        "usd",
        "dollar",
        "gift card"
    ]):
        return "payment_or_charge"

    # 5. RETURN / REFUND
    if contains(t, [
        "refund",
        "refunded",
        "refund pending",
        "refund missing",
        "refund not received",
        "money back",
        "return the item",
        "return item",
        "returning the item",
        "reimburse",
        "reimbursement",
        "easy returns"
    ]):
        return "return_or_refund"

    # 6. PRODUCT
    if contains(t, [
        "kindle",
        "paperwhite",
        "fire tablet",
        "fire tab",
        "echo",
        "alexa",
        "trimmer",
        "product damaged",
        "product broken",
        "item damaged",
        "item broken",
        "stopped charging",
        "water damage",
        "not sealed",
        "unsealed",
        "packaging",
        "outer box",
        "box was the product",
        "defective product",
        "defective item",
        "special offers",
        "special offer",
        "ads on kindle"
    ]):
        return "product_problem"

    # 7. TECHNICAL
    if contains(t, [
        "app",
        "website",
        "browser",
        "error",
        "crash",
        "technical problem",
        "technical issue",
        "not working",
        "isn't working",
        "cannot open",
        "can't open",
        "unable to open",
        "unable to find",
        "can't reach you",
        "cannot reach you",
        "can't reach u",
        "cannot reach u",
        "call back",
        "callback",
        "never received the call",
        "didn't receive the call",
        "did not receive the call"
    ]):
        return "technical_problem"

    # 8. LATE / MISSING DELIVERY
    if contains(t, [
        "late delivery",
        "delivery is late",
        "delivery was late",
        "delivery delay",
        "delivery delayed",
        "package is late",
        "package was late",
        "parcel is late",
        "hasn't arrived",
        "has not arrived",
        "not arrived",
        "haven't received",
        "have not received",
        "never received",
        "still waiting",
        "waiting since",
        "day 5",
        "day 4",
        "nobody has turned up",
        "no one has turned up",
        "no one reached out",
        "delivery attempt",
        "another email about delivery",
        "on time delivery",
        "not delivered",
        "wasn't delivered",
        "was not delivered",
        "deliver tomorrow",
        "delivered tomorrow",
        "need it before",
        "arrive before",
        "delivery problem",
        "stolen package",
        "missing package",
        "missing parcel"
    ]):
        return "late_or_missing_delivery"

    # 9. DELIVERY TRACKING
    if contains(t, [
        "tracking",
        "tracking number",
        "track my order",
        "track order",
        "where is my order",
        "where's my order",
        "delivery status",
        "shipment status",
        "shipping status",
        "amazon logistics",
        "amzl",
        "tba",
        "carrier",
        "delivery date",
        "specific timeline",
        "when will it be delivered",
        "when will my order arrive",
        "who do you use for delivery",
        "who you use for each order",
        "delivery today",
        "multiple products",
        "several products"
    ]):
        return "delivery_tracking"

    # 10. ORDER PROBLEM
    if contains(t, [
        "edit order",
        "edit my order",
        "change my order",
        "change order",
        "order problem",
        "order issue",
        "wrong order",
        "wrong item",
        "preorder",
        "preordered",
        "order says",
        "order status",
        "cancellation email",
        "cancelled email",
        "canceled email",
        "refund email",
        "order email",
        "email containing instructions"
    ]):
        return "order_problem"

    # 11. COMPLAINT / UNRESOLVED
    if contains(t, [
        "not resolved",
        "still unresolved",
        "issue remains",
        "problem remains",
        "not been resolved",
        "not helpful",
        "no response",
        "nobody responded",
        "no one responded",
        "already contacted",
        "already contacted support",
        "contacted several times",
        "contacted multiple times",
        "spoke to several agents",
        "spoke with several agents",
        "not a resolution",
        "look into the matter and solve it",
        "what the hell is going on",
        "huge fail",
        "stop scams",
        "terrible service",
        "awful service",
        "worst service",
        "shameful attitude"
    ]):
        return "complaint_or_unresolved"

    # 12. ML FALLBACK
    X_test = vectorizer.transform([text])
    probabilities = model.predict_proba(X_test)[0]

    best_index = probabilities.argmax()
    prediction = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < 0.45:
        return "other"

    return prediction