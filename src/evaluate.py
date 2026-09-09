import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from intent_classifier import classify_intent

df = pd.read_csv("data/golden_set.csv")

predictions = []

for text in df["text"]:
    predictions.append(classify_intent(text))

df["prediction"] = predictions

accuracy = accuracy_score(df["intent"], df["prediction"])
macro_f1 = f1_score(df["intent"], df["prediction"], average="macro")
weighted_f1 = f1_score(df["intent"], df["prediction"], average="weighted")

print("\n=== INTENT EVALUATION ===")
print("Accuracy:", round(accuracy, 3))
print("Macro F1:", round(macro_f1, 3))
print("Weighted F1:", round(weighted_f1, 3))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    df["intent"],
    df["prediction"],
    zero_division=0
))