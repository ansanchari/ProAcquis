import os
import json
import http.client
from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from crewai.tools import BaseTool
from utils.db import DBManager
from typing import List, Any

class LinkedInProfileCollectorTool(BaseTool):
    name: str = "linkedin_profile_collector"
    description: str = "Collects LinkedIn profile data using RapidAPI"
    
    def _run(self, usernames_str: str) -> str:
        # Parse usernames from string
        usernames = [u.strip() for u in usernames_str.split(",")]
        results = LinkedInDataCollectorAgent.update_profiles(usernames)
        return f"Collected {len(results)} LinkedIn profiles. See details below:\n{results}"

def fetch_linkedin_profile(username, rapidapi_key):
    """Fetch a LinkedIn profile using RapidAPI"""
    print(f"🔍 Attempting to fetch LinkedIn data for: {username}")
    
    conn = http.client.HTTPSConnection("linkedin-data-api.p.rapidapi.com")
    
    headers = {
        'x-rapidapi-key': rapidapi_key,
        'x-rapidapi-host': "linkedin-data-api.p.rapidapi.com"
    }
    
    try:
        # Using the profile endpoint
        conn.request("GET", f"/?username={username}", headers=headers)
        
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        
        if res.status == 200:
            json_data = json.loads(data)
            print(f"✅ SUCCESS: Retrieved LinkedIn profile for {username}")
            return {
                "status": "success",
                "username": username,
                "data": json_data
            }
        else:
            print(f"❌ API ERROR ({res.status}): {data}")
            return {
                "status": "error",
                "username": username,
                "message": f"API returned status {res.status}: {data[:100]}..."
            }
    except Exception as e:
        print(f"❌ EXCEPTION: Failed to retrieve profile for {username}: {str(e)}")
        return {
            "status": "error",
            "username": username,
            "message": f"Exception: {str(e)}"
        }

def store_profile_in_chromadb(profile_data):
    """Store profile data in ChromaDB only"""
    try:
        db_manager = DBManager(path='data/chromadb_data')
        collection = db_manager.get_collection("linkedin_profiles")
        
        # Create text representation of profile for embedding
        if profile_data["status"] == "success":
            profile_text = json.dumps(profile_data["data"])
            username = profile_data["username"]
            
            # Store in ChromaDB
            collection.add(
                documents=[profile_text],
                ids=[username],
                metadatas=[{"name": username, "source": "linkedin"}]
            )
            print(f"✅ Successfully stored {username} profile in ChromaDB")
            return True
        else:
            print(f"⚠️ Skipping ChromaDB storage for {profile_data['username']} due to error status")
            return False
    except Exception as e:
        print(f"❌ ChromaDB Storage Error: {str(e)}")
        return False

class LinkedInDataCollectorAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        # Create tool for the agent
        collector_tool = LinkedInProfileCollectorTool()
        
        return Agent(
            role="LinkedIn Data Collector",
            goal="Fetch detailed candidate profiles from LinkedIn using RapidAPI and store them in ChromaDB",
            backstory="Skilled in interfacing with external APIs, cleaning data and storing profiles efficiently.",
            llm=llm,
            allow_delegation=False,
            tools=[collector_tool]
        )

    @staticmethod
    def update_profiles(usernames):
        rapidapi_key = os.getenv("RAPIDAPI_KEY", "9d945b1d2dmsh17ccfdbee0961bcp11f9dejsn618ca87f16d6")
        collected_profiles = []
        success_count = 0
        error_count = 0
        
        print("\n===== LINKEDIN PROFILE COLLECTION PROCESS =====")
        print(f"Attempting to collect data for {len(usernames)} profiles...")
        
        for username in usernames:
            # Fetch profile data from RapidAPI
            profile_data = fetch_linkedin_profile(username, rapidapi_key)
            collected_profiles.append(profile_data)
            
            # Store in ChromaDB
            if profile_data["status"] == "success":
                stored = store_profile_in_chromadb(profile_data)
                if stored:
                    success_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
        
        # Generate a detailed result summary
        result = f"""
=== LINKEDIN DATA COLLECTION SUMMARY ===
Total profiles attempted: {len(usernames)}
Successfully retrieved and stored: {success_count}
Failed: {error_count}

Detail by profile:
"""
        for profile in collected_profiles:
            if profile["status"] == "success":
                result += f"✅ {profile['username']}: Successfully retrieved and stored in ChromaDB\n"
            else:
                result += f"❌ {profile['username']}: {profile['message']}\n"
                
        return result
