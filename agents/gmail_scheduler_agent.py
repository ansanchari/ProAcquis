from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from crewai.tools import BaseTool
import os
import smtplib
from email.mime.text import MIMEText
from typing import List, Any

smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = os.getenv("GMAIL_SENDER")
sender_password = os.getenv("GMAIL_PASSWORD")

server = smtplib.SMTP(smtp_server, smtp_port)
server.starttls()
server.login(sender_email, sender_password)

def generate_google_meet_link():
    # In a production system, you would call the Google Calendar API to generate a Meet link.
    # For demo purposes, we return a dummy link.
    return "https://meet.google.com/dummy-meet-link"

def send_email(recipient, subject, body):
    # Simple SMTP email sending; ensure you have less-secure apps enabled or use an App Password.
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient

    try:
        server.send_message(msg)
        print(f"✅ Email sent to {recipient}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")
        return False

class EmailSendingTool(BaseTool):
    name: str = "email_scheduler"
    description: str = "Schedules interviews by sending emails with Google Meet links"
    
    def _run(self, emails_str: str) -> str:
        # Parse email addresses from the input string
        emails = [e.strip() for e in emails_str.split(',')]
        results = []
        
        for email in emails:
            try:
                meet_link = generate_google_meet_link()
                subject = "Interview Invitation"
                body = f"""
                Dear Candidate,

                You have been selected for an interview. 
                Please join us using this Google Meet link: {meet_link}

                Best regards,
                HR Team
                """
                success = send_email(email, subject, body)
                if success:
                    results.append(f"✅ Email sent to {email} with meet link: {meet_link}")
                else:
                    results.append(f"❌ Failed to send email to {email}")
            except Exception as e:
                results.append(f"❌ Error sending to {email}: {str(e)}")
        
        return "\n".join(results)

class GmailSchedulerAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        email_tool = EmailSendingTool()
        
        return Agent(
            role="Gmail Scheduler",
            goal="Schedule interviews by generating Google Meet links and sending emails to candidates.",
            backstory="You're responsible for coordinating interview schedules and ensuring candidates receive proper invitations.",
            llm=llm,
            allow_delegation=False,
            tools=[email_tool]
        )

    @staticmethod
    def schedule_interview(candidate_email):
        meet_link = generate_google_meet_link()
        subject = "Interview Invitation"
        body = f"""
        Dear Candidate,

        You have been selected for an interview. 
        Please join us using this Google Meet link: {meet_link}

        Best regards,
        HR Team
        """
        send_email(candidate_email, subject, body)
        return meet_link