import pandas as pd
from escalation import decide_escalation

df = pd.read_csv("data/golden_set.csv")

for _, row in df.iterrows():

    expected = row["escalate"]
    result = decide_escalation(row["text"])
    predicted = "Yes" if result["escalate"] else "No"

    if expected == "Yes" and predicted == "No":
        print("=" * 80)
        print("ID:", row["id"])
        print("CUSTOMER:")
        print(row["text"])
        print("EXPECTED:", expected)
        print("PREDICTED:", predicted)