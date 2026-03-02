import PyPDF2
import plotly.express as px
import streamlit as st
import pandas as pd
import time
import os
import random
import io
from dotenv import load_dotenv
from tasks.hr_tasks import HRTasks
from crewai import Crew, Process
from utils.db import DBManager
from langchain_mistralai import MistralAIEmbeddings
from agents.reporting_agent import ReportingAgent
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
from fpdf import FPDF
import base64

load_dotenv()

st.set_page_config(
    page_title="ProAcquis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /*Main theme colors*/
    :root {
        --primary-color: #6366F1;     
        --secondary-color: #64748B;   
        --accent-color: #10B981;      
        --background-color: #F0FDF4;  
        --text-color: #0F172A;        
        --light-accent: #E2E8F0;      
    }
    
    /*Target Streamlit's main app wrapper for background*/
    .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /*Force header colors*/
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--primary-color) !important;
    }
    
    /*Card styling for dashboard components*/
    div[data-testid="stVerticalBlock"] > div {
        /* This applies a slight card effect to grouped elements */
        border-radius: 8px;
    }
    
    /*Custom button styling - requires !important to override Streamlit*/
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        border-color: var(--primary-color) !important;
    }
    
    .stButton > button:hover {
        background-color: #4F46E5 !important; /* Slightly darker indigo */
        border-color: #4F46E5 !important;
        color: white !important;
    }
    
    /*Chat container styling*/
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        height: 400px;
        overflow-y: auto;
        background-color: white;
    }
    
    /*Dashboard metric styling*/
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: var(--primary-color) !important;
    }
    
    .metric-label {
        color: var(--secondary-color) !important;
    }
    
    /*Chat message styling*/
    .user-message {
        background-color: var(--light-accent);
        color: var(--text-color);
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin: 5px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    
    .agent-message {
        background-color: white;
        color: var(--text-color);
        border: 1px solid var(--light-accent);
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin: 5px 0;
        max-width: 80%;
    }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'should_clear_input' not in st.session_state:
    st.session_state.should_clear_input = False
    
if 'recruitment_data' not in st.session_state:
    st.session_state.recruitment_data = {}
    
if 'job_role' not in st.session_state:
    st.session_state.job_role = ""
    
if 'profiles_loaded' not in st.session_state:
    st.session_state.profiles_loaded = False
    
if 'profiles_found' not in st.session_state:
    st.session_state.profiles_found = False
    
if 'cvs_screened' not in st.session_state:
    st.session_state.cvs_screened = False
    
if 'interviews_scheduled' not in st.session_state:
    st.session_state.interviews_scheduled = False
    
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False
    
if 'final_report' not in st.session_state:
    st.session_state.final_report = ""

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_synthetic_profiles():
    with st.spinner("Loading synthetic profiles from CSV into ChromaDB..."):
        try:
            profiles_df = pd.read_excel("data/cs_engineers.xlsx")
            
            db_manager = DBManager(path='data/chromadb_data')
            try:
                db_manager.client.delete_collection("linkedin_profiles")
            except:
                st.write("No existing collection to delete")
            
            collection = db_manager.get_collection("linkedin_profiles")
            
            embedding_fn = MistralAIEmbeddings(model="mistral-embed", api_key=os.getenv('MISTRAL_API_KEY'))
            
            processed = 0
            for _, row in profiles_df.iterrows():
                try:
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
                    
                    profile_id = f"profile_{row.get('Name', '').lower().replace(' ', '_')}_{processed}"
                    metadata = {
                        "name": str(row.get('Name', 'N/A')),
                        "role": str(row.get('Role', 'N/A')),
                        "location": str(row.get('Location', 'N/A')),
                        "skills": str(row.get('Skills', 'N/A')),
                        "years_experience": str(row.get('Years_of_Experience', 'N/A')),
                        "education": str(row.get('Education', 'N/A'))
                    }
                    
                    collection.add(
                        documents=[profile_text],
                        metadatas=[metadata],
                        ids=[profile_id]
                    )
                    processed += 1
                except Exception as e:
                    st.error(f"Error processing profile {row.get('Name', 'unknown')}: {str(e)}")
            
            st.success(f" Successfully loaded {processed} profiles into ChromaDB")
            st.session_state.profiles_loaded = True
            return processed
        except Exception as e:
            st.error(f"Error loading profiles: {str(e)}")
            return 0

def display_chat_messages():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agent-message">{message["content"]}</div>', unsafe_allow_html=True)

def handle_hr_query(query):
    hr_tasks = HRTasks()
    
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    message_placeholder = st.empty()
    message_placeholder.markdown('<div class="agent-message">Processing your query...</div>', unsafe_allow_html=True)
    
    try:
        response_crew = Crew(
            agents=[hr_tasks.query_response_agent(st.session_state.recruitment_data)],
            tasks=[hr_tasks.answer_hr_query(query, st.session_state.recruitment_data)],
            verbose=True
        )
        
        answer = response_crew.kickoff()
        
        st.session_state.chat_history.append({"role": "assistant", "content": str(answer)})
    except Exception as e:
        error_message = f"I'm sorry, I encountered an issue processing your query. Error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_message})
    
    message_placeholder.empty()
    
    display_chat_messages()
    
def generate_report():
    hr_tasks = HRTasks()
    
    with st.spinner("Generating comprehensive recruitment report..."):
        reporting_crew = Crew(
            agents=[hr_tasks.reporting_agent()],
            tasks=[hr_tasks.generate_report()],
            verbose=True,
            process=Process.sequential
        )
        
        final_report = reporting_crew.kickoff()
        
        st.session_state.recruitment_data["report"] = str(final_report)
        st.session_state.final_report = str(final_report)
        st.session_state.report_generated = True
        
        return final_report

def process_job_role(job_role):
    hr_tasks = HRTasks()
    
    with st.spinner("Interpreting job role..."):
        query_crew = Crew(
            agents=[hr_tasks.hr_query_agent()],
            tasks=[hr_tasks.handle_hr_query(job_role)],
            verbose=True
        )
        
        crew_output = query_crew.kickoff()
        job_details = str(crew_output)
        interpreted_job_role = job_details.strip().replace("Job Role:", "").strip()
        
        st.session_state.recruitment_data["job_role"] = interpreted_job_role
        st.session_state.job_role = interpreted_job_role
        ReportingAgent.add_context('job_role', interpreted_job_role)
        
        return interpreted_job_role

def find_profiles(job_role):
    hr_tasks = HRTasks()
    
    with st.spinner("Searching for matching profiles..."):
        profile_crew = Crew(
            agents=[hr_tasks.profile_finder_agent()],
            tasks=[hr_tasks.find_profiles(job_role)],
            verbose=True
        )
        
        similar_profiles = profile_crew.kickoff()
        
        st.session_state.recruitment_data["profiles"] = str(similar_profiles)
        ReportingAgent.add_context('profiles', str(similar_profiles))
        st.session_state.profiles_found = True
        
        return similar_profiles

def screen_cvs(job_role):
    hr_tasks = HRTasks()
    
    with st.spinner("Screening candidate CVs..."):
        screening_crew = Crew(
            agents=[hr_tasks.cv_screening_agent()],
            tasks=[hr_tasks.screen_cvs(job_role)],
            verbose=True
        )
        
        screened_results = screening_crew.kickoff()
        
        st.session_state.recruitment_data["screening"] = str(screened_results)
        ReportingAgent.add_context('screening', str(screened_results))
        st.session_state.cvs_screened = True
        
        return screened_results

def schedule_interviews():
    hr_tasks = HRTasks()
    
    with st.spinner("Scheduling interviews..."):
        candidate_emails = [" ", " "]
        
        job_role = st.session_state.get("job_role", "Software Engineer")
        
        scheduling_crew = Crew(
            agents=[hr_tasks.gmail_scheduler_agent()],
            tasks=[hr_tasks.schedule_interviews(candidate_emails, job_role=job_role)],
            verbose=True
        )
        
        scheduling_results = scheduling_crew.kickoff()
        
        st.session_state.recruitment_data["scheduling"] = str(scheduling_results)
        ReportingAgent.add_context('scheduling', str(scheduling_results))
        st.session_state.interviews_scheduled = True
        
        return scheduling_results

def export_report_to_pdf(report_text):
    report_str = str(report_text)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Recruitment Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    
    for line in report_str.split('\n'):
        try:
            clean_line = line.encode('latin1', errors='replace').decode('latin1')
            pdf.cell(200, 10, txt=clean_line, ln=True)
        except Exception:
            continue
            
    return pdf.output(dest='S').encode('latin1')

def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href


def process_uploaded_pdfs(uploaded_files):
    with st.spinner(f"Extracting and embedding {len(uploaded_files)} resumes..."):
        processed = 0
        db_manager = DBManager(path='data/chromadb_data')
        collection = db_manager.get_collection("linkedin_profiles")
        
        for file in uploaded_files:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                profile_id = f"pdf_{file.name.replace(' ', '_')}_{random.randint(1000, 9999)}"
                metadata = {
                    "name": file.name.replace('.pdf', ''),
                    "role": "PDF Candidate", 
                    "location": "Unknown",
                    "skills": "Extracted from PDF",
                    "years_experience": "0", 
                    "education": "Extracted from PDF"
                }
                
                collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[profile_id]
                )
                processed += 1
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                
        if processed > 0:
            st.session_state.profiles_loaded = True
        return processed

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/human-resources.png")
    st.title("ProAcquis")
    st.markdown("---")
    
    page = st.radio("Navigate to", ["Dashboard", "Chat Assistant"], index=0)
    
    st.markdown("---")
    
    st.markdown("Upload Resumes")
    uploaded_pdfs = st.file_uploader("Drop candidate PDFs here", type="pdf", accept_multiple_files=True)
    if uploaded_pdfs:
        if st.button("Process Uploaded PDFs"):
            num_processed = process_uploaded_pdfs(uploaded_pdfs)
            st.success(f" Embedded {num_processed} resumes into database!")

    st.markdown("---")
    st.markdown("Workflow Status")
    
    if st.session_state.profiles_loaded:
        st.success("Profiles Database Ready")
    else:
        st.warning("Profiles Not Loaded")
        
    if st.session_state.job_role:
        st.success(f"Job Role: {st.session_state.job_role}")
    else:
        st.warning("No Job Role Defined")
        
    if st.session_state.profiles_found:
        st.success("Matching Profiles Found")
    else:
        st.warning("Profiles Not Searched")
        
    if st.session_state.cvs_screened:
        st.success("CVs Screened")
    else:
        st.warning("CVs Not Screened")
        
    if st.session_state.interviews_scheduled:
        st.success("Interviews Scheduled")
    else:
        st.warning("Interviews Not Scheduled")
        
    if st.session_state.report_generated:
        st.success("Final Report Generated")
    else:
        st.warning("Report Not Generated")
    
def render_analytics_dashboard():
    try:
        db_manager = DBManager(path='data/chromadb_data')
        collection = db_manager.get_collection("linkedin_profiles")
        
        data = collection.get()
        
        if not data or not data['metadatas']:
            st.info("Not enough data to generate analytics yet. Please load profiles.")
            return

        df = pd.DataFrame(data['metadatas'])
        
        df['years_experience'] = pd.to_numeric(df['years_experience'], errors='coerce').fillna(0)

        st.markdown("---")
        st.markdown("Candidate Pool Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_exp = px.histogram(
                df, 
                x="years_experience", 
                nbins=10,
                title="Years of Experience Distribution",
                color_discrete_sequence=['#10B981'] # Emerald Green
            )
            fig_exp.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_exp, use_container_width=True)

        with col2:
            if 'location' in df.columns:
                loc_counts = df['location'].value_counts().reset_index()
                loc_counts.columns = ['Location', 'Count']
                fig_loc = px.pie(
                    loc_counts, 
                    values='Count', 
                    names='Location',
                    title="Candidate Geographic Distribution",
                    hole=0.4, 
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig_loc.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_loc, use_container_width=True)
                
    except Exception as e:
        st.error(f"Could not load analytics: {str(e)}")

if page == "Dashboard":
    st.title("HR Recruitment Dashboard")
    st.markdown("Streamline your recruitment process with AI-powered automation")
    
    if not st.session_state.profiles_loaded:
        if st.button("Load Candidate Database"):
            num_profiles = load_synthetic_profiles()
            if num_profiles > 0:
                st.success(f"Successfully loaded {num_profiles} profiles into the database")
    
    with st.form("job_role_form"):
        job_role_input = st.text_input("Enter Job Role to Search For:", placeholder="e.g., Senior Python Developer")
        submitted = st.form_submit_button("Start Recruitment Process")
        
        if submitted and job_role_input:
            interpreted_job_role = process_job_role(job_role_input)
            st.success(f"Job role interpreted as: {interpreted_job_role}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1. Find Matching Profiles")
        if st.session_state.job_role and not st.session_state.profiles_found:
            if st.button("Find Profiles"):
                similar_profiles = find_profiles(st.session_state.job_role)
                st.success("Matching profiles found!")
                with st.expander("Show Matching Profiles"):
                    st.text(similar_profiles)
        elif st.session_state.profiles_found:
            st.success("Profiles Found")
            with st.expander("Show Matching Profiles"):
                st.text(st.session_state.recruitment_data.get("profiles", "No profiles data available"))
    
    with col2:
        st.markdown("### 2. Screen Candidates")
        if st.session_state.profiles_found and not st.session_state.cvs_screened:
            if st.button("Screen CVs"):
                screened_results = screen_cvs(st.session_state.job_role)
                st.success("CV screening completed!")
                with st.expander("Show Screening Results"):
                    st.text(screened_results)
        elif st.session_state.cvs_screened:
            st.success("Candidates Screened")
            with st.expander("Show Screening Results"):
                st.text(st.session_state.recruitment_data.get("screening", "No screening data available"))
    
    with col3:
        st.markdown("### 3. Schedule Interviews")
        if st.session_state.cvs_screened and not st.session_state.interviews_scheduled:
            if st.button("Schedule Interviews"):
                scheduling_results = schedule_interviews()
                st.success("Interviews scheduled!")
                with st.expander("Show Scheduling Details"):
                    st.text(scheduling_results)
        elif st.session_state.interviews_scheduled:
            st.success("Interviews Scheduled")
            with st.expander("Show Scheduling Details"):
                st.text(st.session_state.recruitment_data.get("scheduling", "No scheduling data available"))


    if st.session_state.profiles_loaded:
        render_analytics_dashboard()

    if st.session_state.interviews_scheduled:

        if st.session_state.interviews_scheduled:
            st.markdown("### Final Report")
            if not st.session_state.report_generated:
                if st.button("Generate Comprehensive Report"):
                    report = generate_report()
                    st.success("Final report generated!")
                    st.markdown("### Recruitment Report")
                    st.text(report)
                    pdf_data = export_report_to_pdf(report)
                    st.markdown(create_download_link(pdf_data, "recruitment_report.pdf"), unsafe_allow_html=True)
            else:
                st.markdown("### Recruitment Report")
                st.text(st.session_state.final_report)
                pdf_data = export_report_to_pdf(st.session_state.final_report)
                st.markdown(create_download_link(pdf_data, "recruitment_report.pdf"), unsafe_allow_html=True)

elif page == "Chat Assistant":
    st.title("HR Assistant Chat")
    st.markdown("Ask questions about candidates, recruitment process, or get specific insights")
    
    if not st.session_state.profiles_loaded:
        st.warning("Please load the candidate database first from the Dashboard")
        if st.button("Load Candidate Database"):
            num_profiles = load_synthetic_profiles()
            if num_profiles > 0:
                st.success(f"Successfully loaded {num_profiles} profiles into the database")
    
    st.markdown("### Chat with HR Assistant")
    
    display_chat_messages()
    
    def set_clear_flag():
        if chat_input.strip():  # Only set flag if there's actual input
            st.session_state.should_clear_input = True
    
    col1, col2 = st.columns([5, 1])
    with col1:
        default_value = "" if st.session_state.get("should_clear_input", False) else st.session_state.get("chat_input", "")
        chat_input = st.text_input("Type your question here:", key="chat_input", 
                                  value=default_value,
                                  placeholder="e.g., Who are the top candidates for the position?")
        if st.session_state.should_clear_input:
            st.session_state.should_clear_input = False
    
    with col2:
        send_button = st.button("Send", on_click=set_clear_flag)
    
    if send_button and chat_input:
        handle_hr_query(chat_input)
    
    st.markdown("### Quick Actions")
    quick_actions = st.columns(3)
    
    with quick_actions[0]:
        if st.button("Get Top Candidates"):
            handle_hr_query("Who are the top candidates for the position?")
    
    with quick_actions[1]:
        if st.button("Generate Report"):
            report = generate_report() if not st.session_state.report_generated else st.session_state.final_report
            handle_hr_query("Show me the comprehensive recruitment report")
    
    with quick_actions[2]:
        if st.button("Candidate Statistics"):
            handle_hr_query("Give me statistics about the candidate pool")

if __name__ == "__main__":
    pass
