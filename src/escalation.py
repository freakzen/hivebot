def decide_escalation(customer_message):

    text = customer_message.lower()
    score = 0
    reasons = []

    strong = [
        "manager",
        "supervisor",
        "escalate",
        "fraud",
        "stolen",
        "scam",
        "legal",
        "consumer court"
    ]

    repeated = [
        "already",
    "before",
    "previously",
    "again",
    "again and again",
    "twice",
    "third time",
    "second time",
    "multiple",
    "several times",
    "repeatedly",
    "keep following up",
    "kept following up"
    ]

    unresolved = [
       "no response",
    "no one",
    "nobody",
    "still waiting",
    "still haven't",
    "still have not",
    "haven't got",
    "have not got",
    "not received",
    "not resolved",
    "still unresolved",
    "same issue",
    "same problem",
    "not helpful",
    "didn't help",
    "did not help",
    "never received",
    "haven't heard",
    "have not heard",
    "no call back",
    "no callback"
    ]

    frustration = [
        "shameful",
        "disgusting",
        "infuriating",
        "worst service",
        "terrible service",
        "awful service",
        "pathetic",
        "useless",
        "incompetent",
        "harassment",
        "fed up",
        "frustrated",
        "disappointed",
        "not good enough"
    ]

    for signal in strong:
        if signal in text:
            score += 3
            reasons.append("high-severity issue")

    for signal in repeated:
        if signal in text:
            score += 2
            reasons.append("repeated support attempt")

    for signal in unresolved:
        if signal in text:
            score += 2
            reasons.append("unresolved support")

    for signal in frustration:
        if signal in text:
            score += 1
            reasons.append("strong frustration")

    if score >= 2:
        return {
            "escalate": True,
            "reason": "Customer shows " + ", ".join(dict.fromkeys(reasons)) + "."
        }

    return {
        "escalate": False,
        "reason": "Issue appears suitable for automated handling."
    }