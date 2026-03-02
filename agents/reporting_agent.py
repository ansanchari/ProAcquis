from crewai import Agent
from langchain_mistralai.chat_models import ChatMistralAI
from utils.db import DBManager
from crewai.tools import BaseTool
import os

class ReportingTool(BaseTool):
    name: str = "reporting_tool"
    description: str = "Generates comprehensive reports based on recruitment data"
    
    def _run(self, query: str = "Generate recruitment report") -> str:
        """Generate a comprehensive recruitment report"""
        return ReportingAgent.generate_report()

class ReportingAgent:
    recruitment_context = {}
    
    @staticmethod
    def add_context(stage, data):
        """Add data to the recruitment context"""
        ReportingAgent.recruitment_context[stage] = data
    
    @staticmethod
    def agent():
        llm = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral/mistral-large-latest"  # Updated to use mistralai provider prefix
        )
        
        report_tool = ReportingTool()
        
        return Agent(
            role="HR Reporting Agent",
            goal="Generate comprehensive recruitment reports with context from all workflow stages.",
            backstory="An efficient summarizer and report generator for HR workflows that integrates recruitment data from multiple sources.",
            llm=llm,
            allow_delegation=False,
            tools=[report_tool]
        )
    
    @staticmethod
    def generate_report():
        """Generate a comprehensive report based on collected context"""
        try:
            db_manager = DBManager(path='data/chromadb_data')
            collection = db_manager.get_collection("linkedin_profiles")
            
            results = collection.query(
                query_texts=["experienced software engineer"],
                n_results=3
            )
            
            report = "=== COMPREHENSIVE RECRUITMENT REPORT ===\n\n"
            
            if 'job_role' in ReportingAgent.recruitment_context:
                report += f"JOB POSITION:\n"
                report += f"Job Role: {ReportingAgent.recruitment_context['job_role']}\n\n"
            
            if 'profiles' in ReportingAgent.recruitment_context:
                report += "CANDIDATE SEARCH RESULTS:\n"
                cleaned_profiles = ReportingAgent.recruitment_context['profiles'].replace("*", "")
                report += cleaned_profiles + "\n\n"
            else:
                if results and results['ids']:
                    report += "SAMPLE CANDIDATES FROM DATABASE:\n"
                    for i in range(len(results['ids'][0])):
                        metadata = results['metadatas'][0][i] if i < len(results['metadatas'][0]) else {}
                        report += f"- {metadata.get('name', 'Unknown')}: {metadata.get('role', 'N/A')}\n"
                    report += "\n"
            
            if 'screening' in ReportingAgent.recruitment_context:
                report += "CV SCREENING RESULTS:\n"
                cleaned_screening = ReportingAgent.recruitment_context['screening'].replace("*", "")
                report += cleaned_screening + "\n\n"
            
            if 'scheduling' in ReportingAgent.recruitment_context:
                report += "INTERVIEW SCHEDULING:\n"
                cleaned_scheduling = ReportingAgent.recruitment_context['scheduling'].replace("*", "")
                report += cleaned_scheduling + "\n\n"
            
            report += "RECOMMENDATIONS:\n"
            
            if 'screening' in ReportingAgent.recruitment_context:
                report += "- Proceed with interviews for recommended candidates\n"
                report += "- Schedule technical assessments for candidates with scores above 70\n"
            else:
                report += "- Further candidate screening recommended\n"
                report += "- Expand search parameters to increase candidate pool\n"
            
            report += "- Consider revisiting job requirements if candidate match rate is low\n\n"
            
            report += "NEXT STEPS:\n"
            report += "1. Conduct interviews with top candidates\n"
            report += "2. Gather feedback from hiring managers\n"
            report += "3. Proceed with reference checks for promising candidates\n"
            report += "4. Prepare offer packages for final candidates\n\n"
            
            report += "SUMMARY:\n"
            if 'screening' in ReportingAgent.recruitment_context and 'profiles' in ReportingAgent.recruitment_context:
                report += "The recruitment process is progressing as expected. "
                report += "Qualified candidates have been identified and evaluated. "
                report += "Proceeding to the interview phase with selected candidates."
            else:
                report += "The recruitment process has been initialized. "
                report += "Candidate search and screening is still in progress. "
                report += "More data is needed to make final recommendations."
                
            return report
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
