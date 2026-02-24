import http.client
import json
import os
from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI


def search_linkedin_profiles(query, api_key):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
      "q": f"{query} site:linkedin.com/in",
      "num": 5  # You can increase this if needed
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    results = json.loads(data.decode("utf-8"))

    linkedin_profiles = []

    for result in results.get("organic", []):
        link = result.get("link", "")
        title = result.get("title", "")
        if "linkedin.com/in/" in link:
            username = link.split("/in/")[1].split("/")[0]
            linkedin_profiles.append({
                "name": title,
                "profile_url": link,
                "username": username
            })

    return linkedin_profiles

class LinkedInSearchAgent:
    @staticmethod
    def agent():
        # The agent only wraps the functionality for logging and describing its role
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        return Agent(
            role="LinkedIn Search Agent",
            goal="Search Google for LinkedIn profiles using SerperAPI.",
            backstory="Efficiently extract LinkedIn usernames by querying Google with a given job search string.",
            llm=llm,
            allow_delegation=False,
            # Here you can bind the search function to the agent if your framework supports it
        )






