HIVEBOT — AI Customer Support Agent
An AI-assisted customer support agent built for the Hiver SDE Intern take-home assignment.
HIVEBOT analyzes incoming customer messages, identifies the customer's intent, retrieves historically similar support interactions, drafts a suitable response, and decides whether the issue should be handled automatically or escalated to a human support agent.
---

USE THIS LINK TO VIEW IT IN ACTION: https://hivebot.streamlit.app/

---
1. Project Overview
Customer support systems receive a large number of messages covering common problems such as:
Delivery delays
Tracking issues
Order problems
Refunds and returns
Payment issues
Account problems
Product problems
Subscription issues
Technical problems
Unresolved complaints
HIVEBOT attempts to automate the first level of support by combining:
Intent classification
Historical support retrieval
Response generation
Escalation decision-making
The system is designed as a practical prototype rather than a production-ready customer support platform.
---
2. System Architecture
The system follows this pipeline:
Customer Message
|
v
+----------------------+
| Intent Classification |
+----------------------+
|
v
+----------------------+
| Historical Retrieval |
+----------------------+
|
v
+----------------------+
| Response Generation  |
+----------------------+
|
v
+----------------------+
| Escalation Decision  |
+----------------------+
|
v
+----------------------+
| Final Agent Output   |
+----------------------+
The final output contains:
Detected intent
Drafted response
Escalation decision
Escalation reason
Historical evidence used by the system
---
3. Why AmazonHelp?
The Customer Support on Twitter dataset contains conversations involving many different companies.
Several brands have large numbers of conversations, including:
AmazonHelp
AppleSupport
Uber_Support
SpotifyCares
Delta
Tesco
AmericanAir
TMobileHelp
AmazonHelp was selected because it provided the largest useful volume of customer-response interactions in the dataset.
The analysis found approximately:
169,840 AmazonHelp brand tweets
100,503 customer replies to AmazonHelp responses
40,671 unique customers
This provided enough historical support data to build a retrieval-based system.
The objective was not to recreate Amazon's actual support system. The objective was to use historical conversations as evidence for designing and evaluating a support-agent prototype.
---
4. Dataset
The project uses the:
Customer Support on Twitter dataset from Kaggle.
Main dataset file:
twcs.csv
The dataset contains approximately 2.8 million tweets.
Important columns include:
Column	Description
tweet_id	Unique tweet identifier
author_id	Twitter account that created the tweet
inbound	Whether the tweet is inbound/customer-side
created_at	Tweet timestamp
text	Tweet text
response_tweet_id	Tweet(s) that this tweet responded to
in_response_to_tweet_id	Tweet being responded to
One important part of working with this dataset was understanding the conversation direction.
For AmazonHelp:
AmazonHelp tweet
|
| response_tweet_id
v
Customer tweet
This relationship was used to construct historical customer-response pairs.
---
5. Historical Support Corpus
After identifying AmazonHelp conversations, the project created a historical corpus containing approximately:
100,503 customer-response pairs
Each pair contains:
Customer message
+
AmazonHelp response
These pairs are used as historical evidence when the system receives a new customer message.
The purpose of retrieval is not to copy a historical answer word-for-word.
Instead, it helps answer:
"How did AmazonHelp historically respond to similar situations?"
This makes the response-generation process more grounded in the selected brand's historical support behavior.
---
6. Intent Classification
The first stage of the system is intent classification.
The final taxonomy contains 12 intents:
```text
delivery_tracking
late_or_missing_delivery
order_problem
cancel_order
return_or_refund
payment_or_charge
account_or_security
product_problem
prime_or_subscription
technical_problem
complaint_or_unresolved
other
```
Intent Definitions
delivery_tracking
Questions about tracking, shipment status, delivery status, carriers, or where an order currently is.
Example:
"Where is my tracking number?"
---
late_or_missing_delivery
Problems where an expected delivery is late, missing, or has not arrived.
Example:
"My package was supposed to arrive yesterday but it hasn't."
---
order_problem
Problems related to placing, modifying, or managing an order.
Example:
"I can't change my order."
---
cancel_order
Requests to cancel an existing order.
Example:
"Please cancel my order."
---
return_or_refund
Returns, refunds, money-back requests, or missing refund issues.
Example:
"I returned the product but haven't received my refund."
---
payment_or_charge
Payment failures, unexpected charges, duplicate charges, or billing-related problems.
Example:
"I was charged twice for the same order."
---
account_or_security
Login, account access, password, verification, or security-related issues.
Example:
"I cannot log into my Amazon account."
---
product_problem
Problems with a physical product, damaged items, defective products, or incorrect product condition.
Example:
"My Kindle stopped charging."
---
prime_or_subscription
Amazon Prime, memberships, subscriptions, benefits, or subscription-related issues.
Example:
"My Prime membership isn't working."
---
technical_problem
Problems with an Amazon application, website, device integration, or technical functionality.
Example:
"The Amazon app keeps crashing."
---
complaint_or_unresolved
Messages primarily focused on unresolved support experiences, repeated complaints, or dissatisfaction without a more specific dominant issue.
Example:
"I contacted support several times and nobody has solved this."
---
other
Messages that do not confidently fit into one of the defined support categories.
---
7. Classifier Development
Several approaches were tested during development.
Baseline 1 — TF-IDF + Logistic Regression
A traditional text classification baseline was created using:
TF-IDF features
Word unigrams and bigrams
Logistic Regression
Initial evaluation:
```text
Accuracy: 32.5%
Macro F1: 0.15
```
---
8. Baseline 2 — Rule-Based Classifier
A rule-based classifier was created using domain-specific keywords and phrases.
This performed better on the small golden evaluation set:
```text
Accuracy: 43.0%
Macro F1: 0.40
Weighted F1: 0.35
```
The rule baseline was useful because customer support messages often contain strong domain-specific signals such as:
refund
tracking
package
cancel
charged
login
Prime
delivery
---
9. Final Classifier
The final classifier uses a hybrid approach:
High-confidence domain rules are checked first.
Machine-learning classification is used as a fallback.
The classifier uses TF-IDF-based text features and Logistic Regression.
The final V5 classifier achieved:
```text
Accuracy: 43.0%
Macro F1: 0.40
```
Per-intent F1 scores:
Intent	F1
account_or_security	0.40
cancel_order	0.33
complaint_or_unresolved	0.22
delivery_tracking	0.48
late_or_missing_delivery	0.39
order_problem	0.36
other	0.55
payment_or_charge	0.11
prime_or_subscription	0.60
product_problem	0.43
return_or_refund	0.67
technical_problem	0.29
The classifier was deliberately kept lightweight and explainable rather than using a large model for every prediction.
---
10. Important Evaluation Caveat
The 200-example golden set was used as the main evaluation dataset.
The labels were created through a combination of initial keyword-based labeling and manual correction during development.
Therefore, the project should not claim that all 200 examples were independently hand-labelled by an external human annotator.
This is an important limitation of the evaluation.
A production version should use independent human annotation with clear annotation guidelines.
---
11. Historical Retrieval
The retrieval system searches the historical AmazonHelp customer-response corpus for messages similar to the incoming customer message.
The current retrieval system uses:
```text
TF-IDF
+
Word unigrams/bigrams
+
Cosine similarity
```
The corpus contains:
```text
100,503 historical customer-response pairs
```
For every incoming message, the system retrieves the top historical examples.
The Streamlit interface displays the top 3 historical examples.
Each example contains:
Historical customer message
Historical AmazonHelp response
Similarity score
---
12. Retrieval Evaluation
A post-filtered retrieval evaluation was performed on the 200-example evaluation set.
Results:
```text
Examples evaluated: 200

Average top-1 similarity: 0.44
Minimum similarity: 0.166
Maximum similarity: 1.0
```
Similarity distribution:
```text
>= 0.2 : 196 / 200 = 98.0%
>= 0.3 : 130 / 200 = 65.0%
>= 0.4 : 79 / 200 = 39.5%
>= 0.5 : 50 / 200 = 25.0%
>= 0.6 : 40 / 200 = 20.0%
```
---
13. What Is Misleading About My Headline Number?
A high retrieval similarity score does not necessarily mean that the retrieved response is useful.
The raw similarity number can be inflated by:
Exact or near-duplicate tweets
Very short messages
Repeated phrases
Shared URLs or usernames
Lexical overlap without semantic similarity
Noisy conversational language
For example, two messages can have a high TF-IDF similarity simply because they contain the same words while actually describing different problems.
Therefore:
```text
Similarity score != Successful support retrieval
```
This is one of the most important lessons from the project.
---
14. Response Generation
The response generation stage uses:
Detected intent
Escalation decision
Historical support evidence
Predefined support-response patterns
The goal is to produce a safe and concise support draft.
Examples include:
Refund
Customer:
"I want my money back."
Possible response:
"We can help with your return or refund. Please check the order's return options and follow the available instructions."
---
Late Delivery
Customer:
"My package is late and I still haven't received it."
Possible response:
"We're sorry your delivery is delayed. Please check the latest tracking information for the order. If the package remains unresolved, a support specialist should review the shipment."
---
Resolved Issue
Customer:
"Everything is good now."
Possible response:
"Glad to hear that! If you need anything else, we're here to help."
---
15. Escalation System
The escalation component is separate from intent classification.
Intent answers:
"What is the customer's problem?"
Escalation answers:
"Does this issue require a human?"
The system looks for signals such as:
Manager or supervisor requests
Fraud or stolen-package reports
Legal threats
Repeated support attempts
Unresolved issues
No response
Still waiting
Multiple previous contacts
Strong frustration
Example:
```text
I asked for a refund twice and haven't got it.
```
Result:
```text
Intent:
return_or_refund

Escalation:
YES

Reason:
Customer shows repeated support attempt.
```
This separation is important because a refund can remain a return_or_refund intent while still requiring human intervention.
---
16. End-to-End Pipeline
The complete pipeline is implemented in:
```text
src/pipeline.py
```
The pipeline performs:
```text
Customer Message
       |
       v
Intent Classification
       |
       v
Escalation Decision
       |
       v
Historical Retrieval
       |
       v
Response Generation
       |
       v
Final Result
```
The output contains:
```text
message
intent
draft_response
escalate
escalation_reason
```
---
17. Streamlit Application
The project includes a Streamlit interface.
The application is called:
```text
HIVEBOT by @freakzen
```
The interface allows the user to enter a customer message and click:
```text
Analyze Message
```
The application then displays:
Agent Decision
Intent
Escalation decision
Historical evidence strength
Draft Response
The proposed support response.
Historical Evidence
The top 3 historically similar AmazonHelp conversations.
Each result shows:
Customer message
AmazonHelp response
Similarity score
---
18. Application UI
The current application also contains links to:
GitHub:
https://github.com/freakzen
LinkedIn:
https://www.linkedin.com/in/zaidchinchali
Contact:
chinchalizaid@gmail.com
Copyright:
© 2026 Zaid Chinchali. All Rights Reserved.
---
19. Evaluation Harness
The project contains evaluation scripts for different components.
Examples include:
```text
src/evaluate_pipeline.py
src/evaluate_retrieval.py
src/evaluate_response_quality.py
src/local_judge.py
src/llm_judge.py
src/evaluate_agreement.py
```
The evaluation system was designed to evaluate:
Intent accuracy
Escalation accuracy
Retrieval quality
Response relevance
Response grounding
Response helpfulness
Response appropriateness
---
20. LLM Judge
An LLM-as-a-judge harness was implemented using the OpenAI API.
The intended evaluation dimensions are:
```text
Relevance
Grounding
Helpfulness
Appropriateness
```
Each dimension is scored from:
```text
1 to 5
```
However, execution of the LLM judge was blocked because the configured API account had exhausted its available API quota.
Therefore, this project does NOT claim fabricated LLM judge scores.
The harness is implemented and can be executed when a valid API quota is available.
---
21. Human Agreement Limitation
A 30-example agreement experiment was also performed.
However, the second set of assessments was generated independently by a model rather than by a human annotator.
Therefore, the results should be interpreted as:
```text
Automated assessment vs independent model assessment
```
and NOT as:
```text
Human agreement
```
True human agreement evidence remains a limitation of this prototype.
A stronger future evaluation would involve:
Independent human annotators
Clear annotation guidelines
Blind evaluation
Cohen's Kappa or similar agreement metrics
---
22. Failure Analysis
The main failure modes observed during evaluation were:
1. Short or Ambiguous Messages
Messages with very little context are difficult to classify.
Example:
```text
Where is it?
```
The system may not know whether the customer is asking about delivery, tracking, an order, or something else.
---
2. Lexical Similarity Without Semantic Similarity
TF-IDF retrieval depends heavily on shared words.
Two messages can share many words while representing different problems.
This can result in irrelevant historical examples.
---
3. Multi-Issue Complaints
Customers often describe several problems in the same message.
Example:
```text
My package is late, I was charged twice, and support hasn't helped me.
```
A single-label classifier must choose one dominant intent even though several issues exist.
---
4. Escalation Situations
Repeated or unresolved cases can retrieve generic support responses even though the customer actually needs human intervention.
This shows why retrieval should not be the only decision mechanism.
---
5. Noisy and Multilingual Tweets
The dataset contains:
Spelling mistakes
Abbreviations
URLs
User mentions
Emojis
Multiple languages
Informal conversational language
Traditional TF-IDF models struggle with these cases.
---
23. Production Risks
This prototype should not automatically send every generated response to customers.
Potential risks include:
Incorrect intent classification
Incorrect retrieval
Outdated historical responses
Sensitive account information
Fraud-related issues
Legal complaints
Refund/payment errors
Ambiguous requests
Multilingual messages
Hallucinated or unsafe responses
A production system should therefore use confidence thresholds and human review.
High-risk cases should always be escalated.
---
24. Future Improvements
Potential improvements include:
Better Intent Classification
Use transformer-based sentence embeddings or a fine-tuned language model.
Better Retrieval
Replace TF-IDF with semantic embeddings and vector search.
Possible architecture:
```text
Customer Message
       |
       v
Embedding Model
       |
       v
Vector Database
       |
       v
Top-K Historical Conversations
```
Better Response Generation
Use an LLM with retrieved historical evidence and strict grounding instructions.
Confidence-Based Routing
Instead of only YES/NO escalation:
```text
High confidence
    -> Auto handle

Medium confidence
    -> Draft + human approval

Low confidence
    -> Human escalation
```
Human Evaluation
Create a larger independently annotated evaluation set.
Multilingual Support
Use multilingual embeddings or multilingual language models.
---
25. Key Lessons
The most important lessons from this project were:
Dataset understanding is critical before model development.
Conversation direction in support datasets can easily be misunderstood.
More data does not automatically produce better intent classification.
Weak supervision can bootstrap a model but introduces noisy labels.
Rule-based systems can be surprisingly competitive on domain-specific support messages.
Retrieval similarity should not be treated as proof of response quality.
Intent and escalation are different problems.
Repeated support attempts are better represented as escalation signals rather than a separate intent.
Evaluation leakage can produce extremely misleading results.
Human evaluation is important for customer-support systems.
A support agent should be conservative when dealing with sensitive or unresolved issues.
A strong headline metric must always be accompanied by its limitations.
---
26. Project Structure
```text
hiver-support-agent/
│
├── app.py
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── download_data.py
│   ├── inspect_data.py
│   ├── find_brands.py
│   ├── brand_analysis.py
│   ├── amazon_analysis.py
│   ├── check_links.py
│   ├── create_intent_sample.py
│   ├── view_sample.py
│   ├── create_training_data_v2.py
│   ├── intent_classifier.py
│   ├── retrieval.py
│   ├── response_generator.py
│   ├── escalation.py
│   ├── pipeline.py
│   ├── evaluate_pipeline.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_response_quality.py
│   ├── local_judge.py
│   ├── llm_judge.py
│   └── evaluate_agreement.py
│
└── data/
    ├── golden_set.csv
    └── twcs.csv
```
The full dataset is intentionally not committed to GitHub because it is large.
---
27. Installation
Clone the repository and enter the project directory.
Install the dependencies:
```powershell
py -m pip install -r requirements.txt
```
---
28. Dataset Setup
The project uses the Customer Support on Twitter dataset.
The downloaded dataset should contain:
```text
twcs/twcs.csv
```
Place the dataset at:
```text
data/twcs.csv
```
The dataset is excluded from Git using `.gitignore`.
---
29. Running HIVEBOT
From the project root:
```powershell
py -m streamlit run app.py
```
The application will start locally.
Open:
```text
http://localhost:8501
```
---
30. Example
Input:
```text
I asked for refund twice and haven't got it.
```
Expected system behavior:
```text
Intent:
return_or_refund

Escalation:
YES

Reason:
Customer shows repeated support attempt.
```
Another example:
```text
My package is late and nobody has contacted me.
```
Expected behavior:
```text
Intent:
late_or_missing_delivery

Escalation:
YES
```
Another example:
```text
I want a refund.
```
Expected behavior:
```text
Intent:
return_or_refund

Escalation:
NO
```
---
31. Design Decisions
Several non-obvious decisions were made during development.
Decision 1 — Select AmazonHelp
AmazonHelp provided the largest useful customer-response volume.
Decision 2 — Use 12 intents
The taxonomy needed enough detail to separate common support problems without creating dozens of classes.
Decision 3 — Separate intent and escalation
A refund can be a normal automated request or a repeated unresolved refund requiring human intervention.
Decision 4 — Use rules before ML
High-confidence domain phrases are often more reliable than weakly trained ML predictions.
Decision 5 — Use historical retrieval
Historical conversations provide evidence of how the selected brand handled similar issues.
Decision 6 — Keep retrieval separate from response generation
Retrieval provides evidence, while response generation determines how that evidence should be used.
Decision 7 — Do not trust raw similarity scores
Similarity can be inflated by duplicates and lexical overlap.
Decision 8 — Avoid evaluation leakage
Training directly on the golden set can produce artificially high metrics.
Decision 9 — Keep the system lightweight
The prototype uses traditional ML and TF-IDF to remain understandable and easy to run locally.
Decision 10 — Escalate high-risk situations
Fraud, stolen packages, legal issues, repeated support attempts, and unresolved cases should receive additional human attention.
Decision 11 — Do not fabricate LLM judge results
The LLM judge was implemented, but its execution was blocked by API quota limitations.
Decision 12 — Be transparent about human evaluation
The independent model comparison was not treated as genuine human agreement.
---
32. Limitations
The current system has several limitations:
The intent classifier has moderate accuracy.
TF-IDF retrieval is lexical rather than semantic.
The response generator is not a fully generative LLM.
Some historical responses may not represent current policies.
The golden dataset requires stronger independent human annotation.
LLM judge execution was blocked by API quota.
Human agreement evidence was not successfully obtained.
Multilingual and highly noisy messages remain difficult.
Multi-intent customer messages are forced into a single dominant intent.
These limitations are intentionally documented rather than hidden.
---
33. Conclusion
HIVEBOT demonstrates a practical approach to building an AI-assisted customer support agent using real historical support conversations.
The system combines:
```text
Intent Classification
        +
Historical Retrieval
        +
Response Drafting
        +
Escalation
```
The project focuses not only on building a working prototype, but also on understanding where such a system can fail.
The main conclusion is that an effective support agent should not rely on a single model or metric.
A practical production system should combine:
```text
Classification
+
Retrieval
+
Confidence
+
Risk Detection
+
Human Escalation
+
Continuous Evaluation
```
HIVEBOT is therefore designed as an AI-assisted support system rather than a fully autonomous replacement for human support agents.
---
Author
Zaid Chinchali
GitHub: https://github.com/freakzen
LinkedIn: https://www.linkedin.com/in/zaidchinchali
Email: chinchalizaid@gmail.com
© 2026 Zaid Chinchali. All Rights Reserved.
