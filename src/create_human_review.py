import pandas as pd

df = pd.read_csv("data/local_judge_results.csv")

sample = df.sample(
    n=min(30, len(df)),
    random_state=42
).copy()

sample["human_relevance"] = ""
sample["human_grounding"] = ""
sample["human_helpfulness"] = ""
sample["human_appropriateness"] = ""
sample["human_reason"] = ""

sample = sample[
    [
        "id",
        "customer",
        "historical_customer",
        "historical_response",
        "similarity",
        "human_relevance",
        "human_grounding",
        "human_helpfulness",
        "human_appropriateness",
        "human_reason"
    ]
]

sample.to_csv(
    "data/human_review.csv",
    index=False
)

print("Created:", len(sample), "human review examples")
print("File: data/human_review.csv")