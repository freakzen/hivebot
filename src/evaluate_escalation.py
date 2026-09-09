import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from escalation import decide_escalation

df = pd.read_csv("data/golden_set.csv")

actual = []
predicted = []

for _, row in df.iterrows():

    result = decide_escalation(row["text"])

    actual.append(1 if row["escalate"] == "Yes" else 0)
    predicted.append(1 if result["escalate"] else 0)

print("\n=== ESCALATION EVALUATION ===")

print("Accuracy:", round(
    accuracy_score(actual, predicted), 3
))

print("Precision:", round(
    precision_score(actual, predicted, zero_division=0), 3
))

print("Recall:", round(
    recall_score(actual, predicted, zero_division=0), 3
))

print("F1:", round(
    f1_score(actual, predicted, zero_division=0), 3
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nActual escalation:", sum(actual))
print("Predicted escalation:", sum(predicted))