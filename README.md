# AI Office Assistant

An autonomous email orchestration assistant powered by Gemini. This agent employs a robust **Plan-Verify-Execute (PVE)** design pattern to ensure email actions are safe, logical, and accurate before they are performed.

## Workflow & Design Architecture

The assistant operates as a modular, agent-based system:

1.  **Plan Phase (`OfficeAgent`):** Translates natural language requests into a structured sequence of discrete tasks (e.g., `check_inbox` -> `generate_email` -> `send_email`) using the Gemini LLM and Pydantic-enforced schemas.
2.  **Verification Phase (`PlanVerifierAgent`):** Acts as an adversarial QA critic. It audits the generated plan for safety, hallucinations, and missing critical parameters, ensuring the execution context is valid.
3.  **Execution Phase (`OfficeAgent` + `EmailManager`):** Sequentially executes the verified tool calls, with dynamic payload resolution to handle data dependencies between steps.

### Key Components

*   **`app.py`**: Streamlit frontend that manages user session state, conversation history, and triggers the PVE loop.
*   **`OfficeAgent.py`**: The central orchestrator. It maps tools to functions, manages the LLM interaction, and handles the stateful planning process.
*   **`PlanVerifierAgent.py`**: Dedicated agent to validate plan logic, safety, and parameter requirements.
*   **`EmailManager.py`**: Encapsulates all Gmail/IMAP/SMTP logic, ensuring isolated and clean communication with the email provider.
*   **`schemas.py`**: Defines Pydantic models for structured tool arguments, plans, and email structures, guaranteeing type-safe interaction between the agent and tools.

---

## Setup Guide

### Prerequisites
*   Python 3.10+
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

    You can set these in your terminal session or a `.env` file:
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
    *   Present its proposed plan for your review.
    *   Automatically verify the plan's logic and safety.
    *   Execute the steps upon successful verification.
    *   Synthesize a final response to inform you of the outcome.

## Safety & Design Principles
*   **Adversarial Planning**: Plans are rejected if they involve synthetic data (fake emails) or unsupported operations.
*   **Context Isolation**: Tools interact with the environment via hardened interfaces in `EmailManager`.
*   **Schema Enforcement**: All tool inputs and outputs are strictly validated against Pydantic models to prevent LLM hallucination and runtime failures.
