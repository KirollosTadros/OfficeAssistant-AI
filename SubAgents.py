import json
from typing import Dict, List, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from tools import TOOLS

INBOX_AGENT_PROMPT = """
You are the Inbox sub-agent of an office assistant.
Use your tools to read the user's inbox or drafts and answer the task you are given.

RULES:
1. Report concrete facts: email id, sender name AND email address, subject, and relevant content.
2. Never invent emails or addresses; only report what the tools return.
3. Finish with a concise, structured summary another agent can act on.
"""

EXPLAIN_AGENT_PROMPT = """
You are the Explain sub-agent of an office assistant. You explain emails to the user in plain language.

RULES:
1. Locate the email: use its id from the task or previous results, otherwise call `check_inbox`
   to find it. Always call `read_email` to read the FULL body before explaining; previews are truncated.
2. Explain, based only on the email's content:
   - Summary: who sent it and what it is about, in 1-2 sentences
   - What the sender wants or is asking for
   - Key details: dates, deadlines, amounts, names, links
   - Action items for the user, and a suggested response if one is needed
3. Point out anything unclear, unusual or suspicious (e.g. urgent payment requests, mismatched sender).
4. Never invent content that is not in the email.
"""

GENERATE_AGENT_PROMPT = """
You are the Generate sub-agent of an office assistant. You compose emails and send them or save them as drafts.

RULES:
1. Only use recipient addresses given in the task or the provided context. Never invent addresses.
2. Compose a clean, professional email:
   - Salutation (e.g. 'Hi [Name],') followed by a blank line
   - Opening/context paragraph, core details paragraph (or bullet points), call to action
   - Sign-off (e.g. 'Best regards,') and the sender name
   Keep distinct paragraphs separated by blank lines; never compress the email into one paragraph.
3. To send, call `send_email`. The user reviews every email before it is sent:
   - If the tool says changes were requested, revise the draft as asked and call `send_email` again.
   - If the tool says the user cancelled, stop and report that the email was not sent.
4. If the task asks for a draft only, call `save_draft` instead of `send_email`.
5. Finish with a one-line report of what happened (sent / saved / cancelled) and to whom.
"""


class SubAgentSpec(TypedDict):
    description: str   # shown to the planner so it can pick the right sub-agent
    prompt: str        # the sub-agent's system prompt
    tools: List[str]   # tool names from tools.TOOLS


# Registry of sub-agents, keyed by agent name. To add a sub-agent, add an entry here.
# The planner prompt, the planner schema and the graph nodes are all built from it.
SUB_AGENTS: Dict[str, SubAgentSpec] = {
    "inbox_agent": {
        "description": "Reads the inbox or saved drafts to find emails: ids, senders, addresses, subjects, content.",
        "prompt": INBOX_AGENT_PROMPT,
        "tools": ["check_inbox", "read_email", "list_drafts"],
    },
    "explain_agent": {
        "description": "Explains an email in plain language: summary, what the sender wants, key details, "
                       "action items and a suggested response. Can find the email in the inbox itself.",
        "prompt": EXPLAIN_AGENT_PROMPT,
        "tools": ["check_inbox", "read_email"],
    },
    "generate_agent": {
        "description": "Composes an email and sends it (the user approves every send) or saves it as a draft.",
        "prompt": GENERATE_AGENT_PROMPT,
        "tools": ["send_email", "save_draft"],
    },
}


def build_sub_agents(model: BaseChatModel) -> Dict:
    """Create one LangChain agent per registry entry, failing fast on unknown tool names."""
    agents = {}
    for name, spec in SUB_AGENTS.items():
        missing = [t for t in spec["tools"] if t not in TOOLS]
        if missing:
            raise ValueError(f"Sub-agent '{name}' uses unknown tools {missing}. Available: {sorted(TOOLS)}")
        agents[name] = create_agent(
            model,
            tools=[TOOLS[t] for t in spec["tools"]],
            system_prompt=spec["prompt"],
            name=name,
        )
    return agents


def describe_sub_agents() -> str:
    """Sub-agent list for the planner prompt."""
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in SUB_AGENTS.items())


def build_task_message(task: dict, request: str, context: List[dict], signature: Optional[str]) -> str:
    """Self-contained brief for a sub-agent: its task plus the outputs of the tasks it depends on."""
    sig_text = (
        f"Sign emails with: {signature}"
        if signature
        else "Signature unknown: use a professional closing without inventing a name."
    )
    parts = [
        f"Original user request: {request}",
        f"Your task: {task['instruction']}",
        sig_text,
    ]
    if context:
        parts.append("Results from previous tasks:\n" + json.dumps(context, indent=2, default=str))
    return "\n\n".join(parts)
