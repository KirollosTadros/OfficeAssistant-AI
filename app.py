import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from OfficeAgent import OfficeAssistant
from SubAgents import SUB_AGENTS

st.set_page_config(page_title="AI Office Assistant", page_icon="✉️", layout="wide")


if "assistant" not in st.session_state:
    st.session_state.assistant = OfficeAssistant()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "signature" not in st.session_state:
    st.session_state.signature = None

assistant: OfficeAssistant = st.session_state.assistant
thread_id: str = st.session_state.thread_id


with st.sidebar:
    st.header("User Profile")
    st.write(f"**Current Signature:** {st.session_state.signature or 'Not set'}")

    manual_sig = st.text_input("Override Signature:", placeholder="e.g., Kirollos Henry")
    if st.button("Update Signature"):
        if manual_sig.strip():
            st.session_state.signature = manual_sig.strip()
            st.success("Signature updated!")
            st.rerun()

    st.divider()
    if st.button("Clear Chat Session", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


def run_graph(updates):
    """Stream graph updates into a status box, then rerun to render replies or pending reviews."""
    with st.chat_message("assistant"):
        with st.status("Working on it...", expanded=True) as status:
            for update in updates:
                for node, value in update.items():
                    if node == "planner" and value and value.get("plan"):
                        status.update(label="Executing plan with sub-agents...")
                        st.write("**Plan:**")
                        for t in value["plan"]:
                            deps = f" _(after {', '.join(t['depends_on'])})_" if t["depends_on"] else ""
                            st.write(f"- `{t['id']}` **{t['agent']}**: {t['title']}{deps}")
                    elif node in SUB_AGENTS:
                        for r in value["results"]:
                            st.write(f"✅ `{r['task_id']}` {r['title']}")
                    elif node == "__interrupt__":
                        status.update(label="Waiting for your approval", state="complete")
                    elif node == "synthesizer":
                        status.update(label="Done", state="complete")
    st.rerun()


st.title("AI Office Assistant")
st.caption("Planner + sub-agents with human-in-the-loop email approval, powered by LangGraph & Gemini")

values = assistant.graph.get_state(assistant.config(thread_id)).values
for msg in values.get("messages", []):
    if isinstance(msg, (HumanMessage, AIMessage)) and msg.text:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.text)


# ---------- Human-in-the-loop email review ----------

pending = assistant.pending_reviews(thread_id)
for intr in pending:
    draft = intr.value["draft"]
    with st.chat_message("assistant"):
        st.markdown("**✉️ Email awaiting your approval**")
        with st.form(key=f"review-{intr.id}"):
            recipient = st.text_input("To", value=draft["recipient"])
            subject = st.text_input("Subject", value=draft["subject"])
            body = st.text_area("Body", value=draft["body"], height=260)
            feedback = st.text_input("Requested changes (for the agent to redraft)")

            approve_col, revise_col, cancel_col = st.columns(3)
            approve = approve_col.form_submit_button("Approve & Send", type="primary")
            revise = revise_col.form_submit_button("Request Changes")
            cancel = cancel_col.form_submit_button("Cancel Email")

    decision = None
    if approve:
        edited = {"recipient": recipient, "subject": subject, "body": body}
        decision = {"action": "edit", "draft": edited} if edited != draft else {"action": "approve"}
    elif revise:
        if feedback.strip():
            decision = {"action": "revise", "feedback": feedback.strip()}
        else:
            st.warning("Describe the changes you want first.")
    elif cancel:
        decision = {"action": "reject"}

    if decision:
        run_graph(assistant.resume(thread_id, {intr.id: decision}))


if prompt := st.chat_input(
    "Approve or change the pending email first" if pending else "How can I assist you with your emails today?",
    disabled=bool(pending),
):
    with st.chat_message("user"):
        st.markdown(prompt)
    run_graph(assistant.chat(thread_id, prompt, signature=st.session_state.signature))
