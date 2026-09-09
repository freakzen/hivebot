from src.retrieval import retrieve


def generate_response(customer_message, intent, escalate=False):
    text = customer_message.lower().strip()

    # No-action / resolved
    resolved_phrases = [
        "everything is good now",
        "all good now",
        "issue is resolved",
        "problem is resolved",
        "resolved now",
        "sorted now",
        "fixed now",
        "thanks for your help",
        "thank you for your help"
    ]

    if any(p in text for p in resolved_phrases):
        return "Glad to hear that! If you need anything else, we're here to help."

    results = retrieve(customer_message, k=5)

    if not results or results[0]["similarity"] < 0.20:
        return (
            "I'm sorry you're experiencing this issue. "
            "Please contact Amazon Customer Service so we can assist you further."
        )

    # Escalated cases
    if escalate:
        if intent == "return_or_refund":
            return (
                "I'm sorry you've had to follow up about this. "
                "Since your refund is still unresolved, this should be "
                "reviewed by a support specialist."
            )

        if intent == "late_or_missing_delivery":
            return (
                "I'm sorry your delivery is still unresolved. "
                "Since you've already had trouble receiving the order, "
                "this should be looked into by a support specialist."
            )

        if intent == "payment_or_charge":
            return (
                "I'm sorry you're still having trouble with this payment issue. "
                "Because it remains unresolved, a support specialist should "
                "review the transaction."
            )

        if intent == "account_or_security":
            return (
                "For your security, this issue should be reviewed directly "
                "by Amazon Customer Service."
            )

        return (
            "I'm sorry you've had trouble getting this resolved. "
            "A support specialist should review the issue further."
        )

    # Normal automated responses
    if intent == "delivery_tracking":
        return (
            "I'm sorry for the inconvenience. "
            "Please check your latest tracking information and delivery status."
        )

    if intent == "late_or_missing_delivery":
        return (
            "I'm sorry for the delay with your delivery. "
            "Please check the latest order status and estimated delivery date."
        )

    if intent == "return_or_refund":
        return (
            "I can help with your return or refund request. "
            "Please check the available return or refund options through "
            "Amazon Customer Service."
        )

    if intent == "cancel_order":
        return (
            "I can help with your cancellation request. "
            "Please check whether the order can still be cancelled through "
            "your Amazon order page."
        )

    if intent == "payment_or_charge":
        return (
            "I'm sorry about the payment issue. "
            "Please check your payment details or contact Amazon Customer "
            "Service so the transaction can be reviewed securely."
        )

    if intent == "account_or_security":
        return (
            "For account security, please use the official Amazon account "
            "support process to resolve this issue."
        )

    if intent == "product_problem":
        return (
            "I'm sorry you're having trouble with the product. "
            "Please contact Amazon Customer Service so we can review the "
            "available options."
        )

    if intent == "prime_or_subscription":
        return (
            "I can help with your Prime or subscription issue. "
            "Please check your Prime membership details and available "
            "account options."
        )

    if intent == "technical_problem":
        return (
            "I'm sorry you're having trouble. "
            "Please let us know what device or app you're using and any "
            "error message you're seeing."
        )

    if intent == "order_problem":
        return (
            "I'm sorry you're having trouble with your order. "
            "Please check the current order details and available options "
            "through Amazon Customer Service."
        )

    return (
        "Thanks for reaching out. "
        "Please provide a little more detail about the issue so we can help."
    )