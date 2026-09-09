import pandas as pd

df = pd.read_csv("data/local_judge_results.csv")

print("\n=== WORST RETRIEVAL CASES ===\n")

df["overall_score"] = (
    df["relevance"]
    + df["grounding"]
    + df["helpfulness"]
    + df["appropriateness"]
) / 4

worst = df.sort_values("overall_score").head(20)

for _, row in worst.iterrows():

    print("=" * 80)

    print("ID:", row["id"])
    print("Overall:", round(row["overall_score"], 2))
    print("Similarity:", round(row["similarity"], 3))

    print("\nCustomer:")
    print(row["customer"])

    print("\nHistorical customer:")
    print(row["historical_customer"])

    print("\nHistorical response:")
    print(row["historical_response"])

    print(
        "\nScores:",
        "Relevance =", row["relevance"],
        "Grounding =", row["grounding"],
        "Helpfulness =", row["helpfulness"],
        "Appropriateness =", row["appropriateness"]
    )