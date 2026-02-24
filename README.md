# ProAcquis: Multi-Agent AI HR Recruitment Pipeline
* **Live Link:** (https://proacquis-areu4ipyqe7pzuaqnq7bqb.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![AI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange)
![DB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)

ProAcquis is an autonomous, AI-driven recruitment dashboard designed to automate the most time-consuming aspects of HR. Instead of relying on manual keyword searches, this application utilizes a custom Retrieval-Augmented Generation (RAG) pipeline to semantically match real PDF resumes to job descriptions. 

## Features
* **PDF Resume Ingestion:** Automatically extracts text from uploaded PDF resumes and embeds them into a local ChromaDB vector database.
* **Semantic Candidate Matching:** Uses Mistral AI embeddings to find candidates based on contextual skill matching.
* **Multi-Agent Orchestration:** Powered by CrewAI, specialized AI agents autonomously screen CVs, debate candidate fit, and schedule mock interviews.
* **Interactive Analytics:** Real-time data visualization of the candidate pool using Plotly.

## Architecture built with:
* **Frontend:** Streamlit 
* **Orchestration:** CrewAI
* **Embeddings & LLM:** Mistral AI 
* **Vector Database:** ChromaDB
* **Data Processing:** PyPDF2, Pandas