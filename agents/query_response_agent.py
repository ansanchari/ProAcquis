import os
from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from utils.db import DBManager
from crewai.tools import BaseTool
from typing import Optional, Dict, Any

class QueryDatabaseTool(BaseTool):
    name: str = "query_database_tool"
    description: str = "Queries the candidate database to answer HR-related questions"
    
    def _run(self, query: str) -> str:
        """Answer HR queries using information from the database"""
        results = QueryResponseAgent.answer_query(query)
        return results

class RetrieveReportTool(BaseTool):
    name: str = "retrieve_report_tool"
    description: str = "Retrieves recruitment report data and statistics"
    
    def _run(self, report_type: str = "full") -> str:
        """Retrieve recruitment report data"""
        results = QueryResponseAgent.get_report_data(report_type)
        return results

class QueryResponseAgent:
    recruitment_data = {}
    
    @staticmethod
    def agent(recruitment_data=None):
        if recruitment_data:
            QueryResponseAgent.recruitment_data = recruitment_data
            
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"
        )
        
        query_tool = QueryDatabaseTool()
        report_tool = RetrieveReportTool()
        
        return Agent(
            role="HR Query Response Agent",
            goal=("Provide accurate and helpful responses to HR queries using information from "
                  "the candidate database and recruitment process data."),
            backstory=("An intelligent assistant specialized in HR recruitment data analysis. "
                      "I can search for candidate information, retrieve reports, and provide "
                      "insights on the recruitment process."),
            llm=llm,
            allow_delegation=False,
            tools=[query_tool, report_tool]
        )

    @staticmethod
    def answer_query(query):
        """Answer HR queries by searching the database and using context"""
        try:
            if QueryResponseAgent.recruitment_data:
                
                if "job_role" in query.lower() and "job_role" in QueryResponseAgent.recruitment_data:
                    return f"Current job role: {QueryResponseAgent.recruitment_data.get('job_role')}"
                    
                if "profiles" in query.lower() and "profiles" in QueryResponseAgent.recruitment_data:
                    return f"Candidate profiles found:\n{QueryResponseAgent.recruitment_data.get('profiles')}"
                    
                if "screen" in query.lower() and "screening" in QueryResponseAgent.recruitment_data:
                    return f"Screening results:\n{QueryResponseAgent.recruitment_data.get('screening')}"
                    
                if "schedule" in query.lower() and "scheduling" in QueryResponseAgent.recruitment_data:
                    return f"Interview scheduling information:\n{QueryResponseAgent.recruitment_data.get('scheduling')}"
            
            db_manager = DBManager(path='data/chromadb_data')
            collection = db_manager.get_collection("linkedin_profiles")
            
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                return "I don't have specific information to answer this query. Please try a different question or provide more context."
            
            response = f"Based on the available information, here's what I found for '{query}':\n\n"
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                
                response += f"--- Candidate {i+1} ---\n"
                if metadata:
                    response += f"Name: {metadata.get('name', 'N/A')}\n"
                    response += f"Role: {metadata.get('role', 'N/A')}\n"
                    response += f"Skills: {metadata.get('skills', 'N/A')}\n"
                    response += f"Experience: {metadata.get('years_experience', 'N/A')}\n"
                
                if i < len(results['documents'][0]):
                    doc_text = results['documents'][0][i][:100] + "..." if len(results['documents'][0][i]) > 100 else results['documents'][0][i]
                    response += f"Profile Summary: {doc_text}\n"
                
                response += "\n"
            
            return response
            
        except Exception as e:
            return f"Error answering query: {str(e)}. Please try a more specific question or check the database connection."

    @staticmethod
    def get_report_data(report_type="full"):
        """Retrieve report data based on the requested type"""
        try:
            if not QueryResponseAgent.recruitment_data:
                return "No recruitment data available for reporting."
            
            if report_type == "summary":
                return "RECRUITMENT SUMMARY:\n" + \
                       f"Job Role: {QueryResponseAgent.recruitment_data.get('job_role', 'Not specified')}\n" + \
                       f"Candidates Found: {len(str(QueryResponseAgent.recruitment_data.get('profiles', '')).split('---')) - 1}\n" + \
                       f"Status: {'Screening completed' if 'screening' in QueryResponseAgent.recruitment_data else 'In progress'}\n" + \
                       f"Interviews: {'Scheduled' if 'scheduling' in QueryResponseAgent.recruitment_data else 'Not yet scheduled'}"
            
            elif report_type == "candidates":
                if "profiles" in QueryResponseAgent.recruitment_data:
                    cleaned_profiles = QueryResponseAgent.recruitment_data.get('profiles').replace("**", "")
                    return f"CANDIDATE PROFILES:\n{cleaned_profiles}"
                else:
                    return "No candidate profiles available yet."
            
            elif report_type == "screening":
                if "screening" in QueryResponseAgent.recruitment_data:
                    cleaned_screening = QueryResponseAgent.recruitment_data.get('screening').replace("**", "")
                    return f"SCREENING RESULTS:\n{cleaned_screening}"
                else:
                    return "No screening results available yet."
            
            else:
                report = "=== RECRUITMENT REPORT ===\n\n"
                
                if "job_role" in QueryResponseAgent.recruitment_data:
                    report += f"JOB ROLE: {QueryResponseAgent.recruitment_data.get('job_role')}\n\n"
                
                if "profiles" in QueryResponseAgent.recruitment_data:
                    report += "CANDIDATE PROFILES:\n"
                    cleaned_profiles = QueryResponseAgent.recruitment_data.get('profiles').replace("**", "")
                    report += cleaned_profiles + "\n\n"
                
                if "screening" in QueryResponseAgent.recruitment_data:
                    report += "SCREENING RESULTS:\n"
                    cleaned_screening = QueryResponseAgent.recruitment_data.get('screening').replace("**", "")
                    report += cleaned_screening + "\n\n"
                
                if "scheduling" in QueryResponseAgent.recruitment_data:
                    report += "INTERVIEW SCHEDULING:\n"
                    cleaned_scheduling = QueryResponseAgent.recruitment_data.get('scheduling').replace("**", "")
                    report += cleaned_scheduling + "\n\n"
                
                return report
                
        except Exception as e:
            return f"Error retrieving report data: {str(e)}"
