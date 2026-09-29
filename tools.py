from langchain_core.tools import tool
from langgraph.types import interrupt

from EmailManager import EmailManager
from schemas import EmailDraft, ReviewDecision

email_manager = EmailManager()


@tool
def check_inbox(max_results: int = 5, unread_only: bool = False) -> list:
    """Fetch the latest inbox messages (id, sender, subject, preview)."""
    return email_manager.check_inbox(max_results=max_results, unread_only=unread_only)


@tool
def read_email(email_id: str) -> dict:
    """Read the full content (sender, recipients, date, subject, body) of one inbox email by its id from check_inbox."""
    return email_manager.read_email(email_id)


@tool
def list_drafts(max_results: int = 5) -> list:
    """List the latest saved Gmail drafts."""
    return email_manager.list_drafts(max_results=max_results)


@tool
def save_draft(recipient: str, subject: str, body: str) -> dict:
    """Save an email to Gmail Drafts without sending it."""
    return email_manager.save_draft(recipient=recipient, subject=subject, body=body)


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email. The draft is shown to the user for approval before it goes out."""
    draft = EmailDraft(recipient=recipient, subject=subject, body=body)

    # Pauses the whole graph until the human resumes with a ReviewDecision.
    decision: ReviewDecision = interrupt({"type": "email_review", "draft": draft.model_dump()})
    action = decision.get("action")

    if action == "edit":
        draft = draft.model_copy(update=decision.get("draft", {}))
        action = "approve"

    if action == "approve":
        email_manager.send_email(**draft.model_dump())
        return f"Email approved by the user and sent to {draft.recipient} (subject: {draft.subject!r})."

    if action == "revise":
        return (
            "The user did NOT send this email and requested changes: "
            f"{decision.get('feedback', '')}\n"
            "Revise the draft accordingly and call send_email again."
        )

    return "The user cancelled this email. Do not send it and do not retry."


# Registry of every tool a sub-agent can use, keyed by tool name.
TOOLS = {t.name: t for t in (check_inbox, read_email, list_drafts, save_draft, send_email)}
