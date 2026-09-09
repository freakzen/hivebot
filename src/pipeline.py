from src.intent_classifier import classify_intent
from src.response_generator import generate_response
from src.escalation import decide_escalation


def run_pipeline(customer_message):
    intent = classify_intent(customer_message)

    escalation = decide_escalation(customer_message)

    response = generate_response(
        customer_message,
        intent,
        escalation["escalate"]
    )

    return {
        "message": customer_message,
        "intent": intent,
        "draft_response": response,
        "escalate": escalation["escalate"],
        "escalation_reason": escalation["reason"]
    }