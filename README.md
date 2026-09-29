# AI Office Assistant

An email orchestration assistant built with **LangChain + LangGraph** and Gemini. A planner agent breaks each request into sub-tasks, fans them out to specialised sub-agents with `Send`, and pauses with `interrupt()` so you approve every email before it is sent.

## Workflow & Design Architecture

```
START -> planner --(clarifying question)--> END
            |
            v
        scheduler --Send(task)--> inbox_agent ----+
            ^     --Send(task)--> explain_agent --+   (independent tasks run in parallel)
            |     --Send(task)--> generate_agent -+
            +-------------------------------------+
            |
            v  (all tasks done)
       synthesizer -> END
```

1.  **Planner:** Gemini with structured output (`Plan`) turns the conversation into `SubTask`s, each assigned to a sub-agent with `depends_on` links. If a recipient or purpose is missing and can't be found in the inbox, it asks the user instead.
2.  **Scheduler + `Send`:** Runs the plan in waves. Every task whose dependencies are done gets its own `Send`, so independent tasks execute in parallel. Each sub-agent receives the outputs of the tasks it depends on.
3.  **Sub-agents** (`langchain.agents.create_agent`), built from the `SUB_AGENTS` registry:
    *   `inbox_agent`: finds emails with `check_inbox`, `read_email`, `list_drafts`
    *   `explain_agent`: explains an email (summary, what the sender wants, key details, action items) with `check_inbox`, `read_email`
    *   `generate_agent`: composes the email, then `send_email` or `save_draft`
4.  **Human-in-the-loop:** `send_email` calls `interrupt()` with the draft. The graph pauses until you resume it with a decision:
    *   **approve**: send as is
    *   **edit**: send your edited recipient / subject / body
    *   **revise**: your feedback goes back to the agent, which redrafts and asks again
    *   **reject**: the email is not sent
5.  **Synthesizer:** Summarises the sub-agent results for the user.

### Adding a sub-agent

Tools and sub-agents are registered in Python dicts, which are the single source of truth. The planner prompt, the planner's structured-output schema (the `agent` field is an enum of the registered names) and the graph nodes are all generated from them:

```python
# tools.py
TOOLS = {t.name: t for t in (check_inbox, read_email, list_drafts, save_draft, send_email)}

# SubAgents.py
SUB_AGENTS = {
    "explain_agent": {
        "description": "Explains an email in plain language ...",  # shown to the planner
        "prompt": EXPLAIN_AGENT_PROMPT,                            # system prompt
        "tools": ["check_inbox", "read_email"],                    # names from TOOLS
    },
    ...
}
```

To add a sub-agent, add one entry to `SUB_AGENTS`; for a new tool, define it with `@tool` and add it to `TOOLS`. An unknown tool name fails at startup.

An `InMemorySaver` checkpointer keeps the conversation per thread and lets interrupted runs resume.

### Key Components

*   **`app.py`**: Streamlit UI that streams the plan's progress and renders an approve / edit / request-changes / cancel form for each paused email.
*   **`OfficeAgent.py`**: The LangGraph orchestrator (`OfficeAssistant`): planner, scheduler, sub-agent workers, synthesizer. It also has a CLI (`python OfficeAgent.py`).
*   **`SubAgents.py`**: The `SUB_AGENTS` registry (description, prompt and tool names per sub-agent) and the builder that turns it into `create_agent` agents.
*   **`tools.py`**: LangChain tools wrapping `EmailManager`, including the `interrupt()`-guarded `send_email`, and the `TOOLS` registry (tool name -> tool).
*   **`EmailManager.py`**: Gmail IMAP/SMTP logic.
*   **`schemas.py`**: The `Plan`/`SubTask` planner schema, the HITL `ReviewDecision`, and the graph state.

---

## Setup Guide

### Prerequisites
*   Python 3.10+ (required by LangChain 1.x)
*   A Google Cloud project with the Gemini API enabled.
*   A Gmail account with an "App Password" enabled for the account.

### Configuration
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set Environment Variables:**
    The application requires the following environment variables to be set:
    *   `GEMINI_API_KEY`: Your Google Gemini API key.
    *   `EMAIL_USER`: Your Gmail email address.
    *   `EMAIL_PASS`: Your Gmail App Password.
    *   `GEMINI_MODEL` (optional): Gemini model name, defaults to `gemini-3.5-flash-lite`.

    You can set these in your terminal session or a `.env` file (loaded automatically):
    ```bash
    export GEMINI_API_KEY="your_api_key"
    export EMAIL_USER="your_email@gmail.com"
    export EMAIL_PASS="your_app_password"
    ```

### Running the Application
```bash
streamlit run app.py
```

---

## Usage

1.  Launch the application using the command above.
2.  The assistant will be available in your browser at the URL provided by Streamlit (usually `http://localhost:8501`).
3.  Type your requests naturally (e.g., "Check my unread emails and draft a reply to the most recent one about the project meeting").
4.  The assistant will:
    *   Show its plan and each sub-agent task as it completes.
    *   Pause and show every email for your approval before it is sent.
    *   Summarise the outcome.

## Safety & Design Principles
*   **Human approval on every send**: `send_email` cannot deliver anything until the graph is resumed with an approval.
*   **No invented recipients**: The planner asks for clarification rather than inventing addresses, and sub-agents only use addresses from the task or the inbox.
*   **Schema enforcement**: Plans and review decisions are typed with Pydantic / TypedDicts.
