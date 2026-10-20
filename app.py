"""College Rules Assistant: pick a policy, read it, and ask questions about it.

Run locally:  streamlit run app.py
"""
import os

import streamlit as st

from core import answer, bedrock, config, retrieve

MAX_QUESTION_CHARS = 500

st.set_page_config(page_title="College Rules Assistant", page_icon="📘", layout="wide")

# On Streamlit Community Cloud, AWS keys come from app secrets instead of a .env file.
for _key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
    try:
        if _key in st.secrets:
            os.environ.setdefault(_key, st.secrets[_key])
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_policy_data(policy_id: str):
    """One policy's index and chunks, loaded once per server process."""
    return retrieve.load(policy_id)


@st.cache_data(show_spinner=False)
def load_pdf(file_name: str) -> bytes:
    return (config.PDF_DIR / file_name).read_bytes()


def ask(policy: dict, question: str) -> dict:
    """Answer one question. Errors are logged and shown to the student as a short message."""
    try:
        index, chunks = load_policy_data(policy["id"])
        return answer.answer_question(
            question, policy, index, chunks,
            embed_fn=bedrock.embed_texts,
            generate_fn=bedrock.generate_text,
        )
    except Exception as exc:
        print(f"[assistant error] policy={policy['id']} error={exc!r}")
        return {
            "answered": False,
            "answer": "Sorry, the assistant could not answer right now. Please try again in a few minutes.",
            "pages": [],
        }


policies = retrieve.list_policies()
st.title("College Rules Assistant")
st.caption("Answers come only from the selected policy and show the page they came from. "
           "This is not an official notice.")

if not policies:
    st.error("No policies are ready yet. Run `python -m core.build_data` to build them.")
    st.stop()

by_id = {p["id"]: p for p in policies}
policy_id = st.selectbox(
    "Policy",
    options=list(by_id),
    format_func=lambda pid: by_id[pid]["title"],
)
policy = by_id[policy_id]

st.caption(
    f"{policy['category']} · Version: {policy['version'] or 'not stated'} · "
    f"Effective: {policy['effective_date'] or 'not stated'} · Owner: {policy['owner_office']}"
)

if "history" not in st.session_state:
    st.session_state.history = {}
history = st.session_state.history.setdefault(policy_id, [])

left, right = st.columns([1.1, 1], gap="large")

with left:
    pdf_path = config.PDF_DIR / policy["file"]
    if pdf_path.exists():
        pdf_bytes = load_pdf(policy["file"])
        st.pdf(pdf_bytes, height=650)
        st.download_button("Download PDF", data=pdf_bytes, file_name=policy["file"], mime="application/pdf")
    else:
        st.error("The PDF for this policy is missing from the server.")

with right:
    st.subheader("Ask a question")
    for item in history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])
            if item["pages"]:
                st.caption("Pages: " + ", ".join(str(p) for p in item["pages"]))

    question = st.chat_input("Type your question about this policy")
    if question:
        question = question.strip()
        if len(question) > MAX_QUESTION_CHARS:
            st.warning(f"Please keep questions under {MAX_QUESTION_CHARS} characters.")
        elif question:
            with st.spinner("Searching the policy…"):
                result = ask(policy, question)
            history.append({"question": question, **result})
            st.rerun()

st.caption("Official notices and the administration office take priority over this assistant. "
           "It does not make disciplinary or legal decisions.")
