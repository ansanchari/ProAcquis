import os
from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from utils.db import DBManager
from crewai.tools import BaseTool
from typing import Optional, Dict, Any

class ProfileSearchTool(BaseTool):
    name: str = "profile_search_tool"
    description: str = "Searches for candidate profiles using similarity search based on a job query"
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Search for profiles matching the query"""
        results = ProfileFinderAgent.search_profiles(query, top_k)
        return results

class ProfileFinderAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        profile_tool = ProfileSearchTool()
        
        return Agent(
            role="Profile Finder",
            goal=("Search the candidate profile database (ChromaDB) using similarity search "
                  "based on a job description and return the most relevant profiles."),
            backstory="Expert at leveraging vector search for recruitment tasks.",
            llm=llm,
            allow_delegation=False,
            tools=[profile_tool]
        )

    @staticmethod
    def search_profiles(query, top_k=5):
        try:
            db_manager = DBManager(path='data/chromadb_data')
            collection = db_manager.get_collection("linkedin_profiles")
            
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results or not results['ids']:
                return "No matching profiles found."
            
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                doc_text = results['documents'][0][i] if i < len(results['documents'][0]) else "No document text available"
                
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                
                profile_info = f"--- Profile {i+1} ---\n"
                
                if metadata:
                    profile_info += f"Name: {metadata.get('name', 'N/A')}\n"
                    profile_info += f"Role: {metadata.get('role', 'N/A')}\n"
                    profile_info += f"Location: {metadata.get('location', 'N/A')}\n"
                    profile_info += f"Skills: {metadata.get('skills', 'N/A')}\n"
                    profile_info += f"Education: {metadata.get('education', 'N/A')}\n"
                    profile_info += f"Years of Experience: {metadata.get('years_experience', 'N/A')}\n"
                
                profile_info += f"\nProfile Details:\n{doc_text}\n"
                
                if 'distances' in results and len(results['distances']) > 0:
                    score = results['distances'][0][i] if i < len(results['distances'][0]) else "N/A"
                    profile_info += f"\nRelevance Score: {score}\n"
                
                formatted_results.append(profile_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching profiles: {str(e)}"
