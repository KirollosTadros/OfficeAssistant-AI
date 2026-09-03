import email
from email.header import decode_header
from email.message import EmailMessage
import imaplib
import os
import smtplib
from typing import Dict, List

def extract_body(raw: email.message.Message) -> str:
    """Helper to extract text/plain body content from an email message."""
    body_content = ""
    if raw.is_multipart():
        for part in raw.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body_content = payload.decode(errors="ignore")
                break
    else:
        payload = raw.get_payload(decode=True)
        if payload:
            body_content = payload.decode(errors="ignore")
    return body_content.strip()


class EmailManager:

    def __init__(self):
        self.EMAIL_USER = os.getenv("EMAIL_USER")
        self.EMAIL_PASS = os.getenv("EMAIL_PASS")

    def _validate_credentials(self):
        if not self.EMAIL_USER or not self.EMAIL_PASS:
            raise ValueError(
                "Email credentials are not configured. Please set the "
                "EMAIL_USER and EMAIL_PASS environment variables."
            )

    def send_email(self, recipient: str, subject: str, body: str) -> Dict:
        self._validate_credentials()
        msg = EmailMessage()
        msg["From"] = self.EMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.EMAIL_USER, self.EMAIL_PASS)
            server.send_message(msg)

        return {"status": "sent", "recipient": recipient}

    def check_inbox(self, max_results: int = 5, unread_only: bool = False, primary: bool = True) -> List[Dict]:
        self._validate_credentials()
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(self.EMAIL_USER, self.EMAIL_PASS)
        mail.select("INBOX")

        if unread_only:
            _, data = mail.search(None, "UNSEEN")
        else:
            _, data = mail.search(None, "ALL")

        mail_ids = data[0].split()
        latest_ids = mail_ids[-max_results:]
        
        messages = []
        for msg_id in reversed(latest_ids):
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            raw = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject_header = raw.get("Subject", "")
            subject, encoding = decode_header(subject_header)[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8", errors="ignore")

            # Extract text snippet using the helper function
            body_text = extract_body(raw)
            body_preview = body_text[:200].replace("\n", " ")

            messages.append({
                "id": msg_id.decode(),
                "from": raw.get("From"),
                "subject": subject,
                "preview": body_preview,
            })

        mail.logout()
        return messages
            
    def save_draft(self, recipient: str, subject: str, body: str) -> Dict:
        self._validate_credentials()
        msg = EmailMessage()
        msg["From"] = self.EMAIL_USER
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(self.EMAIL_USER, self.EMAIL_PASS)
        # Append message directly into Gmail's Drafts mailbox
        mail.append('"[Gmail]/Drafts"', "", imaplib.Time2Internaldate(imaplib.time.time()), msg.as_bytes())
        mail.logout()

        return {"status": "saved", "recipient": recipient, "subject": subject}

    def list_drafts(self, max_results: int = 5) -> List[Dict]:
        self._validate_credentials()
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(self.EMAIL_USER, self.EMAIL_PASS)
        
        # Gmail maps its Drafts label to '[Gmail]/Drafts' over IMAP
        status, _ = mail.select('"[Gmail]/Drafts"')
        if status != "OK":
            mail.logout()
            return [{"error": "Could not open '[Gmail]/Drafts' folder."}]

        _, data = mail.search(None, "ALL")
        mail_ids = data[0].split()
        
        if not mail_ids:
            mail.logout()
            return []

        latest_ids = mail_ids[-max_results:]
        drafts = []

        for msg_id in reversed(latest_ids):
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            raw = email.message_from_bytes(msg_data[0][1])

            # Decode Subject header
            raw_subject = raw.get("Subject", "(No Subject)")
            decoded_subject, encoding = decode_header(raw_subject)[0]
            if isinstance(decoded_subject, bytes):
                decoded_subject = decoded_subject.decode(encoding or "utf-8", errors="ignore")

            drafts.append({
                "id": msg_id.decode(),
                "to": raw.get("To", "(No Recipient)"),
                "subject": decoded_subject,
                "body": extract_body(raw),
            })

        mail.logout()
        return drafts

    
if __name__ == '__main__':
    # Simple check if run directly
    email_manager = EmailManager()
    try:
        emails = email_manager.check_inbox(unread_only=True)
        for mail in emails:
            print(f"Email Subject: {mail.get('subject')}")
    except ValueError as e:
        print(f"Configuration check: {e}")
