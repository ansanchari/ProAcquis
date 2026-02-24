#STRICTLY RESTRICT TO MY DATABASE
import PyPDF2
import random

from dotenv import load_dotenv
from tasks.hr_tasks import HRTasks
from crewai import Crew, Process
import os
import pandas as pd
from utils.db import DBManager
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

def load_synthetic_profiles():
    """Load synthetic profiles from CSV and create embeddings for ChromaDB"""
    print("\n===== LOADING SYNTHETIC PROFILES =====")
    
    #Try multiple approaches to load the Excel file
    profiles_df = None
    approaches = [
        #Approach 1: Standard Excel reading
        lambda: pd.read_excel("data/cs_engineers.xlsx"),
        #Approach 2: Excel with specific sheet index
        lambda: pd.read_excel("data/cs_engineers.xlsx", sheet_name=0),
        #Approach 3: Excel with error handling
        lambda: pd.read_excel("data/cs_engineers.xlsx", na_filter=False),
        #Approach 4: Last resort - try with specific columns
        lambda: pd.read_excel("data/cs_engineers.xlsx", 
                             names=['Name', 'Role', 'Location', 'Skills', 'Years_of_Experience', 
                                   'Achievements', 'Education', 'Certifications'])
    ]
    
    for i, approach in enumerate(approaches):
        try:
            profiles_df = approach()
            print(f"Successfully loaded {len(profiles_df)} profiles using approach #{i+1}")
            break
        except Exception as e:
            print(f"Approach #{i+1} failed: {str(e)}")
            
    if profiles_df is None:
        print("All parsing attempts failed. Could not load profiles.")
        return 0
    
    #Clean up the data if needed
    try:
        #Ensure all expected columns exist
        required_columns = ['Name', 'Role', 'Location', 'Skills', 'Years_of_Experience', 
                           'Achievements', 'Education', 'Certifications']
        for col in required_columns:
            if col not in profiles_df.columns:
                print(f"Missing column: {col}")
                
        #Convert Years_of_Experience to numeric
        if 'Years_of_Experience' in profiles_df.columns:
            profiles_df['Years_of_Experience'] = pd.to_numeric(profiles_df['Years_of_Experience'], errors='coerce')
    except Exception as e:
        print(f"Error during data cleanup: {str(e)}")
    
    #Clear existing ChromaDB collection
    db_manager = DBManager(path='data/chromadb_data')
    try:
        db_manager.client.delete_collection("linkedin_profiles")
        print("Deleted existing ChromaDB collection")
    except:
        print("No existing collection to delete")
    
    #Create a new collection
    collection = db_manager.get_collection("linkedin_profiles")
    
    #Create embeddings using MistralAI
    embedding_fn = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv('MISTRAL_API_KEY'))
    
    #Process each profile and add to ChromaDB
    processed = 0
    for _, row in profiles_df.iterrows():
        try:
            #Create a text representation of the profile
            profile_text = f"""
            Name: {row.get('Name', 'N/A')}
            Role: {row.get('Role', 'N/A')}
            Location: {row.get('Location', 'N/A')}
            Skills: {row.get('Skills', 'N/A')}
            Years of Experience: {row.get('Years_of_Experience', 'N/A')}
            Achievements: {row.get('Achievements', 'N/A')}
            Education: {row.get('Education', 'N/A')}
            Certifications: {row.get('Certifications', 'N/A')}
            """
            
            #Create a unique ID for the profile
            profile_id = f"profile_{row.get('Name', '').lower().replace(' ', '_')}_{processed}"
            
            #Metadata for the profile
            metadata = {
                "name": str(row.get('Name', 'N/A')),
                "role": str(row.get('Role', 'N/A')),
                "location": str(row.get('Location', 'N/A')),
                "skills": str(row.get('Skills', 'N/A')),
                "years_experience": str(row.get('Years_of_Experience', 'N/A')),
                "education": str(row.get('Education', 'N/A'))
            }
            
            #Add to ChromaDB with embeddings
            collection.add(
                documents=[profile_text],
                metadatas=[metadata],
                ids=[profile_id]
            )
            processed += 1
        except Exception as e:
            print(f"Error processing profile {row.get('Name', 'unknown')}: {str(e)}")
    
    print(f"✅ Successfully loaded {processed} profiles into ChromaDB")
    return processed

def process_uploaded_pdfs(pdf_file_paths):
    """CLI version: Provide a list of file paths (strings) to process PDFs"""
    print(f"\n===== EXTRACTING AND EMBEDDING {len(pdf_file_paths)} RESUMES =====")
    processed = 0
    db_manager = DBManager(path='data/chromadb_data')
    collection = db_manager.get_collection("linkedin_profiles")
    
    for file_path in pdf_file_paths:
        try:
            #1. Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            #Extract just the filename from the path for the ID
            file_name = os.path.basename(file_path)
            
            #2. Create basic metadata
            profile_id = f"pdf_{file_name.replace(' ', '_')}_{random.randint(1000, 9999)}"
            metadata = {
                "name": file_name.replace('.pdf', ''),
                "role": "PDF Candidate", 
                "location": "Unknown",
                "skills": "Extracted from PDF",
                "years_experience": "0", 
                "education": "Extracted from PDF"
            }
            
            #3. Add to vector database
            collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[profile_id]
            )
            processed += 1
            print(f"  - Successfully processed: {file_name}")
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
            
    print(f"✅ Successfully loaded {processed} PDF profiles into ChromaDB")
    return processed

def main():
    #Store all recruitment data in a dictionary to pass to the query agent
    recruitment_data = {}
    
    hr_query = input("HR, please enter your job-role query: ")

    hr_tasks = HRTasks()

    #Step 1: Interpret HR's query first.
    query_crew = Crew(
        agents=[hr_tasks.hr_query_agent()],
        tasks=[hr_tasks.handle_hr_query(hr_query)],
        verbose=True
    )

    crew_output = query_crew.kickoff()
    job_details = str(crew_output)
    job_role = job_details.strip().replace("Job Role:", "").strip()
    
    print(f"Interpreted job role: {job_role}")
    
    #Store job role in recruitment data and reporting context
    recruitment_data["job_role"] = job_role
    from agents.reporting_agent import ReportingAgent
    ReportingAgent.add_context('job_role', job_role)

    #Load synthetic data instead of LinkedIn API
    print("\nLoading synthetic profiles from CSV into ChromaDB...")
    num_profiles = load_synthetic_profiles()
    print(f"Loaded {num_profiles} synthetic profiles into the database.")
    
    #Step 2: Retrieve relevant profiles using RAG & similarity search.
    profile_crew = Crew(
        agents=[hr_tasks.profile_finder_agent()],
        tasks=[hr_tasks.find_profiles(hr_query)],
        verbose=True
    )
    similar_profiles = profile_crew.kickoff()
    print("Similar profiles retrieved:")
    print(similar_profiles)
    
    #Store profiles in recruitment data and reporting context
    recruitment_data["profiles"] = str(similar_profiles)
    ReportingAgent.add_context('profiles', str(similar_profiles))

    #Step 3: Screen CVs among the retrieved profiles.
    screening_crew = Crew(
        agents=[hr_tasks.cv_screening_agent()],
        tasks=[hr_tasks.screen_cvs(job_role)],
        verbose=True
    )
    screened_results = screening_crew.kickoff()
    print("Screened CV results:")
    print(screened_results)
    
    #Store screening results in recruitment data and reporting context
    recruitment_data["screening"] = str(screened_results)
    ReportingAgent.add_context('screening', str(screened_results))

    #Step 4: Schedule interviews using Gmail Scheduler. 
    #For demo purposes, use a predefined list of candidate emails.
    candidate_emails = ["", ""]
    scheduling_crew = Crew(
        agents=[hr_tasks.gmail_scheduler_agent()],
        tasks=[hr_tasks.schedule_interviews(candidate_emails, job_role=job_role)],
        verbose=True
    )
    scheduling_results = scheduling_crew.kickoff()
    print("Scheduling results:")
    print(scheduling_results)
    
    #Store scheduling results in recruitment data and reporting context
    recruitment_data["scheduling"] = str(scheduling_results)
    ReportingAgent.add_context('scheduling', str(scheduling_results))

    #Step 5: Generate the final report.
    reporting_crew = Crew(
        agents=[hr_tasks.reporting_agent()],
        tasks=[hr_tasks.generate_report()],
        verbose=True,
        process=Process.sequential
    )
    final_report = reporting_crew.kickoff()
    print("Final Recruitment Report:")
    print(final_report)
    
    #Store the final report in recruitment data
    recruitment_data["report"] = str(final_report)

    #Interactive HR Query Loop
    print("\n\n===== HR Interactive Query Mode =====")
    print("You can now ask questions about the recruitment process, candidates, or reports.")
    print("Type 'exit' to quit.")
    
    while True:
        hr_question = input("\nEnter your question: ")
        if hr_question.lower() in ['exit', 'quit', 'q']:
            print("Exiting HR Query Mode. Goodbye!")
            break
        
        #Create a crew to answer the specific HR query
        response_crew = Crew(
            agents=[hr_tasks.query_response_agent(recruitment_data)],
            tasks=[hr_tasks.answer_hr_query(hr_question, recruitment_data)],
            verbose=True
        )
        
        answer = response_crew.kickoff()
        print("\n----- Answer -----")
        print(str(answer))

if __name__ == "__main__":
    main()
