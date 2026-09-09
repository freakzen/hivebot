import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from intent_classifier import classify_intent


df = pd.read_csv("data/golden_set.csv")

predictions = []

for text in df["text"]:
    predictions.append(
        classify_intent(text)
    )

df["prediction"] = predictions

accuracy = accuracy_score(
    df["intent"],
    df["prediction"]
)

print("\n=== INTENT CLASSIFIER V2 ===")

print(
    "Accuracy:",
    round(accuracy * 100, 1),
    "%"
)

print("\nClassification Report:\n")

print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        df["intent"],
        df["prediction"]
    )
)

df.to_csv(
    "data/intent_v2_results.csv",
    index=False
)

print("\nSaved:")
print("data/intent_v2_results.csv")