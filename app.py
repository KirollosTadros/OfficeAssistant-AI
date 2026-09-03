import streamlit as st
import json
from OfficeAgent import InteractiveOfficeAssistant

st.set_page_config(page_title="AI Office Assistant", page_icon="✉️", layout="wide")


if "assistant" not in st.session_state:
    st.session_state.assistant = InteractiveOfficeAssistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "waiting_for_signature" not in st.session_state:
    st.session_state.waiting_for_signature = False



with st.sidebar:
    st.header("User Profile")
    saved_sig = st.session_state.assistant.user_profile.get("signature_name") or "Not set"
    st.write(f"**Current Signature:** {saved_sig}")
    
    manual_sig = st.text_input("Override Signature:", placeholder="e.g., Kirollos Henry")
    if st.button("Update Signature"):
        if manual_sig.strip():
            st.session_state.assistant.user_profile["signature_name"] = manual_sig.strip()
            st.success("Signature updated!")
            st.rerun()

    st.divider()
    if st.button("Clear Chat Session", type="primary"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.waiting_for_signature = False
        st.rerun()



st.title("Autonomous Office Assistant")
st.caption("Plan-Verify-Execute Email Orchestrator powered by Gemini")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "plan" in msg:
            with st.expander("View Execution Plan"):
                for task in msg["plan"]:
                    st.write(f"**Step {task.step_id}:** {task.title}")



if prompt := st.chat_input("How can I assist you with your emails today?"):
    assistant = st.session_state.assistant

    if st.session_state.waiting_for_signature:
        assistant.user_profile["signature_name"] = prompt.strip()
        st.session_state.waiting_for_signature = False

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(f"User: {prompt}")
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Thinking and planning...", expanded=True) as status:
            
            # 1. PLAN PHASE
            status.update(label="1/3: Formulating plan...")
            plan_obj = assistant.generate_plan(st.session_state.conversation_history)
            
            st.write("**Generated Plan:**")
            for task in plan_obj.plan:
                st.write(f"- **Step {task.step_id}:** {task.title}")

            # 2. VERIFICATION PHASE
            status.update(label="2/3: Verifying plan safety & arguments...")
            verification = assistant.verifier.verify_plan(
                "\n".join(st.session_state.conversation_history),
                plan_obj,
                known_signature=assistant.user_profile.get("signature_name")
            )

            if not verification.is_valid:
                status.update(label="⚠️ Verification required clarification", state="error")
                clarification = verification.question_for_user or "Could you clarify the missing details?"
                
                if any(k in clarification.lower() for k in ["name", "signature", "who is sending"]):
                    st.session_state.waiting_for_signature = True

                st.markdown(clarification)
                st.session_state.messages.append({"role": "assistant", "content": clarification, "plan": plan_obj.plan})
                st.session_state.conversation_history.append(f"Assistant: {clarification}")
                st.stop()

            status.update(label="3/3: Executing verified tasks...", state="running")
            execution_context = []

            for task in plan_obj.plan:
                if task.tool_name in ["final_response", "ask_user"]:
                    continue

                st.write(f"⚙️ Running **Step {task.step_id}:** {task.title}...")
                func = assistant.tool_map[task.tool_name]
                args = assistant.get_args(task.arguments)

                if task.tool_name == "generate_email" and execution_context:
                    args["context"] = f"{args.get('context', '')}\nPrior outputs: {json.dumps(execution_context, default=str)}"
                elif task.tool_name == "send_email":
                    draft = next((item["output"] for item in reversed(execution_context) if item["tool"] == "generate_email"), None)
                    if draft and isinstance(draft, dict):
                        args.update(draft)

                result = func(**args)
                execution_context.append({
                    "step_id": task.step_id,
                    "tool": task.tool_name,
                    "output": result
                })

            status.update(label="All steps completed successfully!", state="complete")

        synthesis_prompt = f"""
        User Query: {prompt}
        Execution Results: {json.dumps(execution_context, default=str)}
        Summarize outcome concisely.
        """
        summary = assistant.client.models.generate_content(
            model=assistant.model_name,
            contents=synthesis_prompt
        )

        st.markdown(summary.text)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": summary.text,
            "plan": plan_obj.plan
        })
        st.session_state.conversation_history.append(f"Assistant: {summary.text}")