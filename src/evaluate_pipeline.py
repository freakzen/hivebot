import pandas as pd
from pipeline import run_pipeline

df = pd.read_csv("data/golden_set.csv")

results = []

for _, row in df.iterrows():

    result = run_pipeline(row["text"])

    expected_escalate = str(row["escalate"]).strip().lower() == "yes"

    results.append({
        "id": row["id"],
        "customer": row["text"],
        "expected_intent": row["intent"],
        "predicted_intent": result["intent"],
        "expected_escalate": expected_escalate,
        "predicted_escalate": result["escalate"],
        "escalation_reason": result["escalation_reason"],
        "draft_response": result["draft_response"]
    })

output = pd.DataFrame(results)

output.to_csv(
    "data/end_to_end_results.csv",
    index=False
)

intent_accuracy = (
    output["expected_intent"]
    == output["predicted_intent"]
).mean()

escalation_accuracy = (
    output["expected_escalate"]
    == output["predicted_escalate"]
).mean()

print("\n=== END-TO-END EVALUATION ===")

print("Examples:", len(output))

print(
    "Intent accuracy:",
    round(intent_accuracy * 100, 1),
    "%"
)

print(
    "Escalation accuracy:",
    round(escalation_accuracy * 100, 1),
    "%"
)

print("\nSaved:")
print("data/end_to_end_results.csv")