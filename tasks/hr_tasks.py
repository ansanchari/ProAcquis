from crewai import Agent, Task
from agents.cv_screening_agent import CVScreeningAgent
from agents.reporting_agent import ReportingAgent
from agents.hr_query_agent import HRQueryAgent
from agents.linkedin_search_agent import LinkedInSearchAgent, search_linkedin_profiles
from agents.linkedin_data_collector_agent import LinkedInDataCollectorAgent
from agents.profile_finder_agent import ProfileFinderAgent
from agents.gmail_scheduler_agent import GmailSchedulerAgent
from agents.query_response_agent import QueryResponseAgent

class HRTasks:
    def hr_query_agent(self):
        return HRQueryAgent.agent()

    def cv_screening_agent(self):
        return CVScreeningAgent.agent()

    def reporting_agent(self):
        return ReportingAgent.agent()

    def linkedin_search_agent(self):
        return LinkedInSearchAgent.agent()

    def linkedin_data_collector_agent(self):
        return LinkedInDataCollectorAgent.agent()

    def profile_finder_agent(self):
        return ProfileFinderAgent.agent()

    def gmail_scheduler_agent(self):
        return GmailSchedulerAgent.agent()

    def query_response_agent(self, recruitment_data):
        return QueryResponseAgent.agent(recruitment_data)

    # Existing task to handle HR query.
    def handle_hr_query(self, hr_query):
        return Task(
            description=(f"Interpret this HR query: '{hr_query}'. Clearly specify the exact job role and essential skills. "
                         "Pass these details to subsequent tasks."),
            agent=self.hr_query_agent(),
            expected_output="Identified job role and essential skills."
        )

    # Task for running the LinkedIn search (to be executed occasionally for database population)
    def run_linkedin_search(self, query):
        return Task(
            description=f"Search for LinkedIn usernames using query: '{query}'.",
            agent=self.linkedin_search_agent(),
            expected_output="List of LinkedIn usernames."
        )

    # Task for populating the database by collecting detailed data
    def populate_database(self, usernames):
        # Convert usernames list to string for task description
        usernames_str = ", ".join(usernames) if isinstance(usernames, list) else str(usernames)
        
        return Task(
            description=f"Update the database with LinkedIn profiles for these usernames: {usernames_str}",
            agent=self.linkedin_data_collector_agent(),
            expected_output="Confirmation that the profiles were successfully fetched and stored."
        )

    # Task for finding profiles from the persistent store using similarity search
    def find_profiles(self, job_description):
        return Task(
            description=f"Search the candidate profiles in the database that are similar to: '{job_description}'.",
            agent=self.profile_finder_agent(),
            expected_output="List of similar candidate profiles."
        )

    # Updated task for scheduling interviews using Gmail integration.
    def schedule_interviews(self, candidate_emails, job_role="Software Engineer"):
        # Convert the list of emails to a comma-separated string
        emails_str = ", ".join(candidate_emails) if isinstance(candidate_emails, list) else str(candidate_emails)
        
        # Create the task with job_role in description
        return Task(
            description=f"Schedule interviews for the '{job_role}' position with the following candidates: {emails_str}. Send them personalized emails with Google Meet links. Job role: {job_role}",
            agent=self.gmail_scheduler_agent(),
            expected_output="Confirmation that interview invitations were sent to all candidates."
        )

    def screen_cvs(self, job_role):
        return Task(
            description=f"Screen candidate CVs for the '{job_role}' position. Evaluate technical skills, experience, and education. Rank candidates based on their suitability.",
            agent=self.cv_screening_agent(),
            expected_output="Ranked list of candidates with evaluation scores and comments."
        )

    def generate_report(self):
        return Task(
            description="Generate a comprehensive report on the recruitment process, candidate evaluations, and recommendations.",
            agent=self.reporting_agent(),
            expected_output="Detailed recruitment report with insights and recommendations."
        )

    def answer_hr_query(self, query, recruitment_data):
        return Task(
            description=f"Answer the following HR query: '{query}'. Provide detailed information based on the available recruitment data.",
            agent=self.query_response_agent(recruitment_data),
            expected_output="A comprehensive answer to the HR query with relevant information from the recruitment process."
        )

