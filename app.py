import streamlit as st
import sys

sys.path.append("src")

from pipeline import run_pipeline
from retrieval import retrieve


st.set_page_config(
    page_title="HiveBot By Freakzen",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Header
# -----------------------------

st.title("HIVEBOT by @freakzen")

st.caption(
    "An Intent-Classifier based model which provides better outcomes for User Queries"
)

st.divider()


# -----------------------------
# Customer Message
# -----------------------------

message = st.text_area(
    "Customer message",
    placeholder="Example: My order hasn't arrived and the tracking hasn't updated.",
    height=120
)


# -----------------------------
# Analyze Message
# -----------------------------

if st.button("Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing customer message..."):

        result = run_pipeline(message)
        historical = retrieve(message, k=3)


    # -----------------------------
    # Agent Decision
    # -----------------------------

    st.subheader("Agent Decision")

    col1, col2, col3 = st.columns(3)


    # Intent
    with col1:

        st.markdown("### Intent")

        st.info(result["intent"])


    # Escalation
    with col2:

        st.markdown("### Escalation")

        if result["escalate"]:
            st.error("YES")
        else:
            st.success("NO")


    # Evidence
    with col3:

        st.markdown("### Evidence")

        if historical:

            best_similarity = historical[0]["similarity"]

            if best_similarity >= 0.5:
                st.success("Strong")

            elif best_similarity >= 0.3:
                st.warning("Moderate")

            else:
                st.error("Weak")

            st.write(
                f"Best similarity: {best_similarity:.3f}"
            )

        else:

            st.error("No evidence")


    # -----------------------------
    # Escalation Reason
    # -----------------------------

    st.caption(
        f"Escalation reason: {result['escalation_reason']}"
    )


    st.divider()


    # -----------------------------
    # Draft Response
    # -----------------------------

    st.subheader("Draft Response")

    st.text_area(
        "AI-generated response",
        value=result["draft_response"],
        height=160
    )


    st.divider()


    # -----------------------------
    # Historical Evidence
    # -----------------------------

    st.subheader("Historical Evidence")

    if not historical:

        st.warning(
            "No sufficiently similar historical conversations were found."
        )


    for i, item in enumerate(historical, 1):

        similarity = item["similarity"]

        with st.expander(
            f"Example {i} — similarity {similarity:.3f}"
        ):

            st.markdown("**Historical customer:**")

            st.write(
                item["customer"]
            )

            st.markdown("**Historical Amazon response:**")

            st.write(
                item["amazon"]
            )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.markdown(
    "### Connect with me"
)

col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    st.link_button(
        "🐙 GitHub",
        "https://github.com/freakzen"
    )

with col2:
    st.link_button(
        "💼 LinkedIn",
        "https://www.linkedin.com/in/zaidchinchali"
    )

st.markdown("---")

st.caption(
    "© 2026 Zaid Chinchali. All Rights Reserved."
)

st.caption(
    "Contact: chinchalizaid@gmail.com"
)