import pandas as pd
from sklearn.metrics import cohen_kappa_score

automated = pd.read_csv("data/local_judge_results.csv")
independent = pd.read_csv("data/human_review_labeled_independent.csv")

df = automated.merge(
    independent[
        [
            "id",
            "human_relevance",
            "human_grounding",
            "human_helpfulness",
            "human_appropriateness"
        ]
    ],
    on="id",
    how="inner"
)

metrics = [
    ("relevance", "human_relevance"),
    ("grounding", "human_grounding"),
    ("helpfulness", "human_helpfulness"),
    ("appropriateness", "human_appropriateness")
]

print("\n=== AUTOMATED vs INDEPENDENT AGREEMENT ===")

for metric, independent_col in metrics:

    automated_scores = df[metric]
    independent_scores = df[independent_col]

    agreement = (
        automated_scores == independent_scores
    ).mean() * 100

    kappa = cohen_kappa_score(
        automated_scores,
        independent_scores
    )

    print(f"\n{metric.capitalize()}")
    print("Exact agreement:", round(agreement, 1), "%")
    print("Cohen's kappa:", round(kappa, 3))

print("\nExamples compared:", len(df))