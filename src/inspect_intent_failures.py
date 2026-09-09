import pandas as pd

df = pd.read_csv("data/intent_v2_results.csv")

failures = df[
    df["intent"] != df["prediction"]
].copy()

print("\n=== INTENT CLASSIFIER FAILURES ===")

print("Total failures:", len(failures))

print("\nMost common confusion:")

print(
    failures.groupby(
        ["intent", "prediction"]
    ).size()
    .sort_values(ascending=False)
    .head(20)
)

print("\nTop 30 individual failures:\n")

for _, row in failures.head(30).iterrows():

    print("=" * 80)

    print("ID:", row["id"])

    print("EXPECTED:", row["intent"])

    print("PREDICTED:", row["prediction"])

    print("\nCUSTOMER:")

    print(row["text"])