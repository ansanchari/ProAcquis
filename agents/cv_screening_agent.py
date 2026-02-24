#PROFILE SCREENING

from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from utils.db import DBManager
from crewai.tools import BaseTool
import os
from typing import Optional, Dict, Any

class CVSearchTool(BaseTool):
    name: str = "cv_search_tool"
    description: str = "Searches and screens candidate profiles based on job requirements"
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Search for profiles and screen them based on job requirements"""
        results = CVScreeningAgent.search_and_screen_profiles(query, top_k)
        return results

class CVScreeningAgent:
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        # Create the CV search and screening tool
        cv_tool = CVSearchTool()
        
        return Agent(
            role="CV Screener",
            goal=("Screen and score CVs according to job suitability by analyzing candidate "
                  "profiles from the database and evaluating their skills and experience."),
            backstory="Highly skilled in analyzing CVs swiftly and accurately.",
            llm=llm,
            allow_delegation=False,
            tools=[cv_tool]
        )
    
    @staticmethod
    def search_and_screen_profiles(job_description, top_k=5):
        try:
            # First, search for relevant profiles using similarity search
            db_manager = DBManager(path='data/chromadb_data')
            collection = db_manager.get_collection("linkedin_profiles")
            
            # Use ChromaDB's query function for similarity search
            results = collection.query(
                query_texts=[job_description],
                n_results=top_k
            )
            
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                return "No matching profiles found in the database."
            
            # Now screen and evaluate the retrieved profiles
            screened_results = []
            
            for i in range(len(results['ids'][0])):
                # Access document text and metadata
                doc_text = results['documents'][0][i] if i < len(results['documents'][0]) else "No document text available"
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                
                # Extract relevant skills from job description (simplified)
                job_keywords = job_description.lower().split()
                
                # Calculate matching score based on skills and experience
                score = 0
                skills = metadata.get('skills', '').lower()
                years_exp = metadata.get('years_experience', '0')
                
                # Convert years experience to number if possible
                try:
                    years_exp = float(years_exp)
                except:
                    years_exp = 0
                
                # Score based on experience (0-40 points)
                experience_score = min(40, int(years_exp * 8))
                
                # Score based on skill match (0-60 points)
                skill_score = 0
                for keyword in job_keywords:
                    if keyword in skills and len(keyword) > 3:  # Avoid matching short words
                        skill_score += 5
                skill_score = min(60, skill_score)
                
                # Total score
                score = experience_score + skill_score
                
                # Format evaluation - ONLY using data from the database
                evaluation = f"--- Candidate {i+1}: {metadata.get('name', 'Unknown')} ---\n"
                
                # Only include fields that exist in the metadata
                for field in ['role', 'location', 'skills', 'years_experience', 'education']:
                    if field in metadata and metadata[field]:
                        field_name = field.replace('_', ' ').title()
                        evaluation += f"{field_name}: {metadata[field]}\n"
                
                # Add evaluation scores
                evaluation += f"\nEvaluation:\n"
                evaluation += f"Experience Score: {experience_score}/40\n"
                evaluation += f"Skills Match Score: {skill_score}/60\n"
                evaluation += f"Overall Score: {score}/100\n"
                
                # Add recommendation based on score
                if score >= 80:
                    recommendation = "Highly Recommended"
                elif score >= 60:
                    recommendation = "Recommended"
                elif score >= 40:
                    recommendation = "Consider for Interview"
                else:
                    recommendation = "Not Recommended"
                
                evaluation += f"Recommendation: {recommendation}\n"
                
                # Store the complete evaluation with the score
                screened_results.append((score, evaluation))
            
            # Sort results by score (highest first)
            screened_results.sort(key=lambda x: x[0], reverse=True)
            
            # Format the final output
            final_output = "=== CV SCREENING RESULTS ===\n\n"
            final_output += f"Screened {len(screened_results)} candidates from the database for job: {job_description}\n\n"
            final_output += "Candidates Ranked by Suitability (DATABASE PROFILES ONLY):\n\n"
            
            for i, (score, evaluation) in enumerate(screened_results):
                final_output += f"Rank #{i+1} (Score: {score}/100)\n"
                final_output += f"{evaluation}\n\n"
            
            # Add an explicit disclaimer
            final_output += "DISCLAIMER: All profile information above comes directly from the database. No profile data has been generated or modified."
            
            return final_output
            
        except Exception as e:
            return f"Error screening profiles: {str(e)}"
