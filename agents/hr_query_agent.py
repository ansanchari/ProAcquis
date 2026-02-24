from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
import os
from tenacity import retry, stop_after_attempt, wait_exponential

class HRQueryAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest",
            temperature=0.3,  # Lower temperature for more consistent responses
            max_retries=5,  # Add retry mechanism for API calls
            retry_min_seconds=4,
            retry_max_seconds=60
        )
        return Agent(
            role="HR Query Handler",
            goal="Interpret HR's job role queries to instruct other agents.",
            backstory=(
                "You are an intelligent HR assistant capable of interpreting HR's natural language queries about recruitment requirements. "
                "You clearly identify the requested job role and skills and instruct other agents accordingly."
            ),
            llm=llm,
            allow_delegation=True,
            max_rpm=5,  # Limit requests per minute
            max_execution_time=300  # Allow more time for retries
        )
