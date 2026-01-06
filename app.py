# import os
# import uuid
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from typing import TypedDict, Optional, List
# from langgraph.graph import StateGraph, END
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# # --- SET PAGE CONFIG FIRST ---
# st.set_page_config(layout="wide")

# # --- 1. Load API Key ---
# load_dotenv()
# if not os.environ.get("GROQ_API_KEY"):
#     st.error("GROQ_API_KEY not found in .env file. Please add it to run the app.")
#     st.stop()

# # --- 2. Define Agentic Logic ---

# # Define the LLM for the Screening Agent
# # Note: I changed the model back to llama3, as 'openai/gpt-oss-20b'
# # is not a standard Groq model and may cause errors.
# llm = ChatGroq(
#     model="openai/gpt-oss-20b",
#     api_key=os.environ.get("GROQ_API_KEY"),
#     temperature=0.0
# )

# # Define the JSON structure for the screening report
# class ScreeningReport(TypedDict):
#     score: int
#     key_strengths: List[str]
#     potential_gaps: List[str]
#     suggested_interview_questions: List[str]
#     summary: str

# def run_screening_agent(state: dict) -> dict:
#     """Runs the LLM-powered screening agent."""
#     st.session_state.log.append("--- 🧠 Running Screening Agent ---")
    
#     job_description = state["job_description"]
#     resume = state["resume"]

#     prompt_template = """
#     You are an expert HR screening agent. Your task is to analyze the
#     provided resume against the job description.

#     Return *only* a JSON object that matches the following structure:
#     {{
#         "score": <int, 0-100 score of relevance>,
#         "key_strengths": [<string, specific examples from resume that match JD>],
#         "potential_gaps": [<string, skills from JD missing in resume>],
#         "suggested_interview_questions": [<string, questions to ask based on gaps>],
#         "summary": "<string, a 2-sentence summary of the candidate's fit>"
#     }}

#     Job Description:
#     {job_description}

#     Candidate Resume:
#     {resume}
#     """

#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     parser = JsonOutputParser(pydantic_object=ScreeningReport)
#     chain = prompt | llm | parser

#     try:
#         report = chain.invoke({
#             "job_description": job_description,
#             "resume": resume
#         })
        
#         st.session_state.log.append(f"--- ✅ Screening Report Generated (Score: {report['score']}) ---")
        
#         return {
#             "screening_report": report,
#             "candidate_status": "screening_passed"
#         }
#     except Exception as e:
#         st.session_state.log.append(f"--- ❌ Error in Screening Agent: {e} ---")
#         return {
#             "candidate_status": "screening_failed",
#             "screening_report": {"summary": f"Failed to parse resume: {e}", "score": 0}
#         }

# # Stubbed Agents
# def run_scheduling_agent(state: dict) -> dict:
#     st.session_state.log.append("--- 🗓️ Running Scheduling Agent (Simulation) ---")
#     simulated_schedule = "2025-11-05 at 10:00 AM"
#     st.session_state.log.append(f"--- ✅ Simulated interview booked for {simulated_schedule} ---")
#     return {
#         "interview_schedule": simulated_schedule,
#         "candidate_status": "interview_scheduled"
#     }

# def run_document_agent(state: dict) -> dict:
#     st.session_state.log.append("--- 📄 Running Document Agent (Simulation) ---")
#     simulated_url = "https://docusign.com/offer-letter-xyz123"
#     st.session_state.log.append(f"--- ✅ Simulated offer letter sent ---")
#     return {
#         "offer_letter_url": simulated_url,
#         "candidate_status": "offer_sent"
#     }

# def run_compliance_agent(state: dict) -> dict:
#     st.session_state.log.append("--- 🛡️ Running Compliance Agent (Simulation) ---")
#     st.session_state.log.append(f"--- ✅ Simulated background check passed. IT provisioning initiated. ---")
#     return {
#         "compliance_status": "passed",
#         "candidate_status": "hired"
#     }

# def handle_rejection(state: dict) -> dict:
#     st.session_state.log.append("--- ⛔ Handling Rejection ---")
#     score = state.get("screening_report", {}).get("score", "N/A")
#     st.session_state.log.append(f"--- ⛔ Candidate rejected. (Score: {score}) ---")
#     return {
#         "candidate_status": "rejected"
#     }

# # --- 3. Define Graph Logic ---

# class AgentState(TypedDict):
#     job_description: str
#     resume: str
#     candidate_status: Optional[str]
#     screening_report: Optional[ScreeningReport]
#     interview_schedule: Optional[str]
#     offer_letter_url: Optional[str]
#     compliance_status: Optional[str]

# def decide_after_screening(state: AgentState) -> str:
#     st.session_state.log.append("--- 🚦 Deciding next step after screening ---")
#     score = state.get("screening_report", {}).get("score", 0)
#     if score >= 70:
#         st.session_state.log.append(f"--- ✅ Score is {score}. Proceeding to Scheduling. ---")
#         return "scheduling"
#     else:
#         st.session_state.log.append(f"--- ⛔ Score is {score}. Proceeding to Rejection. ---")
#         return "rejection"

# @st.cache_resource
# def build_graph():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("screening", run_screening_agent)
#     workflow.add_node("scheduling", run_scheduling_agent)
#     workflow.add_node("document", run_document_agent)
#     workflow.add_node("compliance", run_compliance_agent)
#     workflow.add_node("rejection", handle_rejection)
#     workflow.set_entry_point("screening")
#     workflow.add_conditional_edges(
#         "screening",
#         decide_after_screening,
#         {"scheduling": "scheduling", "rejection": "rejection"}
#     )
#     workflow.add_edge("scheduling", "document")
#     workflow.add_edge("document", "compliance")
#     workflow.add_edge("compliance", END)
#     workflow.add_edge("rejection", END)
#     return workflow.compile()

# app = build_graph()

# # --- 4. Streamlit UI ---

# st.title("🤖 Agentic AI HR Recruitment System")
# st.markdown("This interface allows you to test the multi-agent recruitment pipeline visually.")

# if "log" not in st.session_state:
#     st.session_state.log = []
# if "final_state" not in st.session_state:
#     st.session_state.final_state = None
# if "run_complete" not in st.session_state:
#     st.session_state.run_complete = False

# # --- Sidebar for Inputs ---
# with st.sidebar:
#     st.header("Inputs")
    
#     default_jd = """Job Title: Senior Python Developer
# Description: We are seeking a Senior Python Developer with 8+ years
# of experience. Must be proficient in Django, PostgreSQL, and AWS
# (S3, EC2, Lambda). Experience with NoSQL databases like MongoDB
# and testing frameworks like Pytest is a major plus."""
    
#     jd_input = st.text_area("Job Description", value=default_jd, height=200)
    
#     st.divider()
    
#     st.subheader("Candidate Resume")
    
#     uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    
#     default_resume_text = """John Doe | Senior Software Engineer
# Experience: 10 years
# Summary: Senior developer skilled in Python, Django, and AWS.
# Skills:
# - Python, Django, Flask
# - PostgreSQL, MongoDB, Redis
# - AWS (EC2, S3, Lambda, SQS)
# - Pytest, Unittest
# - Docker, Kubernetes
# """
#     resume_input_text = st.text_area("Or Paste Resume Text", value=default_resume_text, height=300)
    
#     resume_content = ""
#     if uploaded_file is not None:
#         try:
#             if uploaded_file.type == "application/pdf":
#                 reader = PdfReader(uploaded_file)
#                 for page in reader.pages:
#                     resume_content += page.extract_text()
#                 st.sidebar.success("PDF Resume Parsed!")
#             elif uploaded_file.type == "text/plain":
#                 resume_content = uploaded_file.getvalue().decode("utf-8")
#                 st.sidebar.success("TXT Resume Parsed!")
#         except Exception as e:
#             st.sidebar.error(f"Error parsing file: {e}")
#             resume_content = ""
    
#     if not resume_content:
#         resume_content = resume_input_text 

#     if st.button("Run Recruitment Pipeline"):
#         if not resume_content.strip():
#             st.sidebar.error("Please upload or paste a resume.")
#         else:
#             st.session_state.log = []
#             st.session_state.final_state = None
#             st.session_state.run_complete = False
            
#             inputs = {
#                 "job_description": jd_input,
#                 "resume": resume_content
#             }
            
#             config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
#             with st.spinner("Recruitment pipeline in progress..."):
#                 final_state = {}
#                 for event in app.stream(inputs, config=config, stream_mode="values"):
#                     final_state = event 
                    
#                     if "candidate_status" in final_state:
#                         last_log = st.session_state.log[-1] if st.session_state.log else ""
#                         new_status_log = f"--- 🛰️ Status updated to: {final_state['candidate_status']} ---"
#                         if new_status_log != last_log:
#                             st.session_state.log.append(new_status_log)
                
#                 st.session_state.final_state = final_state
#                 st.session_state.run_complete = True
#                 st.session_state.log.append("--- 🎉 Pipeline Finished ---")
            
#             st.success("Pipeline finished!")

# # --- Main Page for Results ---
# if st.session_state.run_complete and st.session_state.final_state:
    
#     state = st.session_state.final_state
#     report = state.get("screening_report")

#     # --- Section 1: AI Screening Report ---
#     if report:
#         st.header("AI Screening Report")
        
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.metric("Overall Score", f"{report.get('score', 0)}/100")
#             st.subheader("Final Status")
            
#             status = state.get('candidate_status', 'N/A').upper()
#             if status == "REJECTED":
#                 st.error(status)
#             elif status == "HIRED":
#                 st.success(status)
#             else:
#                 st.info(status)

#         with col2:
#             st.subheader("Screening Summary")
#             st.write(report.get('summary', 'No summary provided.'))
        
#         st.divider()

#         # --- Show Screening Details only if not rejected ---
#         if state.get("candidate_status") != "rejected":
#             st.subheader("Key Strengths")
#             strengths = report.get('key_strengths', [])
#             if strengths:
#                 st.dataframe(strengths, use_container_width=True, column_config={"value": "Strength"})
            
#             st.subheader("Potential Gaps & Verification Points")
#             gaps = report.get('potential_gaps', [])
#             if gaps:
#                 st.dataframe(gaps, use_container_width=True, column_config={"value": "Gap"})
                
#             st.subheader("AI-Suggested Interview Questions")
#             questions = report.get('suggested_interview_questions', [])
#             if questions:
#                 st.code('\n'.join(f"{i+1}. {q}" for i, q in enumerate(questions)), language='markdown')
            
#     else:
#         st.error("Screening failed to produce a report.")
    
#     st.divider()

#     # --- NEW: Section 2: Full Pipeline Status ---
#     st.header("Pipeline Agent Status")
    
#     # Scheduling Agent
#     if state.get("interview_schedule"):
#         with st.expander("🗓️ Scheduling Agent", expanded=True):
#             st.success("Interview Scheduled")
#             st.info(f"Simulated meeting booked for: {state['interview_schedule']}")
#             st.markdown("**(Next Step:** Replace this stub with a real Google Calendar API call.)")
    
#     # Document Agent
#     if state.get("offer_letter_url"):
#         with st.expander("📄 Document Agent", expanded=True):
#             st.success("Offer Letter Sent")
#             st.info("Simulated offer letter generated and sent via DocuSign.")
#             st.code(f"Mock URL: {state['offer_letter_url']}", language='text')
#             st.markdown("**(Next Step:** Replace this stub with a real DocuSign API call.)")

#     # Compliance Agent
#     if state.get("compliance_status") == "passed":
#         with st.expander("🛡️ Compliance Agent", expanded=True):
#             st.success("Compliance Passed")
#             st.info("Simulated background check cleared and IT accounts provisioned.")
#             st.markdown("**(Next Step:** Replace this stub with real Checkr or Google Workspace API calls.)")
    
#     # Rejection
#     if state.get("candidate_status") == "rejected":
#         with st.expander("⛔ Rejection", expanded=True):
#             st.error("Candidate Rejected")
#             st.info(f"Reason: Low screening score ({report.get('score', 'N/A')}).")
#             st.markdown("**(Next Step:** This agent could be expanded to send a polite rejection email.)")

#     # --- Section 3: Execution Log ---
#     st.divider()
#     st.subheader("Pipeline Execution Log")
#     st.text_area("Log", value='\n'.join(st.session_state.log), height=300, disabled=True)

# else:
#     st.info("Upload or paste a resume in the sidebar and click 'Run' to start the pipeline.")









import os
import uuid
import streamlit as st
import requests
import json
from docusign_esign import ApiClient, EnvelopesApi, EnvelopeDefinition, TemplateRole
from datetime import datetime, timedelta, timezone  # <-- ADDED timezone
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(layout="wide")

# --- 1. Load API Key ---
load_dotenv()
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file. Please add it to run the app.")
    st.stop()

# --- 2. Define Agentic Logic ---

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.0
)

class ScreeningReport(TypedDict):
    score: int
    first_name: str
    last_name: str
    email: str
    key_strengths: List[str]
    potential_gaps: List[str]
    suggested_interview_questions: List[str]
    summary: str

def run_screening_agent(state: dict) -> dict:
    st.session_state.log.append("--- 🧠 Running Screening Agent ---")
    
    job_description = state["job_description"]
    resume = state["resume"]

    prompt_template = """
    You are an expert HR screening agent. Your task is to analyze the
    provided resume against the job description. Extract the candidate's
    full name and email.

    Return *only* a JSON object that matches the following structure:
    {{
        "score": <int, 0-100 score of relevance>,
        "first_name": "<string, candidate's first name>",
        "last_name": "<string, candidate's last name>",
        "email": "<string, candidate's email address>",
        "key_strengths": [<string, specific examples from resume that match JD>],
        "potential_gaps": [<string, skills from JD missing in resume>],
        "suggested_interview_questions": [<string, questions to ask based on gaps>],
        "summary": "<string, a 2-sentence summary of the candidate's fit>"
    }}

    Job Description:
    {job_description}

    Candidate Resume:
    {resume}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser(pydantic_object=ScreeningReport)
    chain = prompt | llm | parser

    try:
        report = chain.invoke({
            "job_description": job_description,
            "resume": resume
        })
        
        st.session_state.log.append(f"--- ✅ Screening Report Generated (Score: {report['score']}) ---")
        
        return {
            "screening_report": report,
            "candidate_status": "screening_passed"
        }
    except Exception as e:
        st.session_state.log.append(f"--- ❌ Error in Screening Agent: {e} ---")
        return {
            "candidate_status": "screening_failed",
            "screening_report": {"summary": f"Failed to parse resume: {e}", "score": 0}
        }

# --- FINAL FIX: Real Scheduling Agent (Calendly) ---
def run_real_scheduling_agent(state: dict) -> dict:
    """
    Uses the Calendly API to schedule an invite.
    This version correctly calculates the start time in the local timezone.
    """
    st.session_state.log.append("--- 🗓️ Running REAL Scheduling Agent (Calendly) ---")
    
    try:
        report = state["screening_report"]
        first_name = report["first_name"]
        last_name = report["last_name"]
        email = report["email"]
        
        token = os.environ.get("CALENDLY_TOKEN")
        event_type_url = os.environ.get("CALENDLY_EVENT_TYPE_URL")

        url = "https://api.calendly.com/invitees"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # --- HERE IS THE FIX ---
        
        # 1. Define the target timezone (Asia/Calcutta is UTC+5:30)
        tz_calcutta = timezone(timedelta(hours=5, minutes=30))
        
        # 2. Get the current date and set the desired local time (e.g., 2 PM)
        # We'll use 4 days from now to be safe (e.g., Fri -> Tue)
        target_day = datetime.now(tz_calcutta) + timedelta(days=4)
        start_time_local = target_day.replace(hour=14, minute=0, second=0, microsecond=0)
        
        # 3. Convert the local start time to UTC for the API
        start_time_utc = start_time_local.astimezone(timezone.utc)

        # Payload structure
        payload = {
            "event_type": event_type_url,
            # Pass the correct UTC time to the API
            "start_time": start_time_utc.isoformat(), 
            "invitee": {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "timezone": "Asia/Calcutta" # Let Calendly know the invitee's timezone
            },
            "location": { 
                "kind": "google_conference"
            }
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))

        if response.status_code == 201:
            invitee_uri = response.json().get("resource", {}).get("uri")
            st.session_state.log.append(f"--- ✅ Calendly invite created: {invitee_uri} ---")
            return {
                "interview_schedule": invitee_uri,
                "candidate_status": "interview_scheduled"
            }
        else:
            # Log the detailed error from Calendly
            st.session_state.log.append(f"--- ❌ Calendly Error: {response.text} ---")
            return {"candidate_status": "scheduling_failed"}

    except Exception as e:
        st.session_state.log.append(f"--- ❌ Scheduling Agent Error: {e} ---")
        return {"candidate_status": "scheduling_failed"}

# --- Real Document Agent (DocuSign) ---
def run_real_document_agent(state: dict) -> dict:
    """
    Uses the DocuSign API (JWT Grant) to send a document from a template.
    """
    st.session_state.log.append("--- 📄 Running REAL Document Agent (DocuSign) ---")
    
    try:
        report = state["screening_report"]
        candidate_email = report["email"]
        candidate_name = f"{report['first_name']} {report['last_name']}"
        
        client_id = os.environ.get("DOCUSIGN_INTEGRATION_KEY")
        account_id = os.environ.get("DOCUSIGN_ACCOUNT_ID")
        user_id = os.environ.get("DOCUSIGN_USER_ID")
        base_path = os.environ.get("DOCUSIGN_BASE_PATH")
        template_id = os.environ.get("DOCUSIGN_TEMPLATE_ID")
        private_key_file = os.environ.get("DOCUSIGN_PRIVATE_KEY_FILE")
        
        # This is the line that fails if the file is missing
        with open(private_key_file, "r") as f:
            private_key = f.read()

        api_client = ApiClient()
        api_client.host = base_path
        
        token = api_client.request_jwt_user_token(
            client_id=client_id,
            user_id=user_id,
            oauth_host_name="account-d.docusign.com", 
            private_key_bytes=private_key,
            expires_in=3600,
            scopes=["signature", "impersonation"]
        )
        api_client.set_default_header("Authorization", f"Bearer {token.access_token}")

        template_role = TemplateRole(
            email=candidate_email,
            name=candidate_name,
            role_name="Candidate" # MUST match the role name in your DocuSign Template
        )
        
        envelope_definition = EnvelopeDefinition(
            template_id=template_id,
            template_roles=[template_role],
            status="sent"
        )
        
        envelopes_api = EnvelopesApi(api_client)
        results = envelopes_api.create_envelope(account_id, envelope_definition=envelope_definition)
        envelope_id = results.envelope_id
        
        st.session_state.log.append(f"--- ✅ DocuSign envelope sent: {envelope_id} ---")
        
        return {
            "offer_letter_url": f"Envelope ID: {envelope_id}",
            "candidate_status": "offer_sent"
        }

    except Exception as e:
        st.session_state.log.append(f"--- ❌ DocuSign Agent Error: {e} ---")
        return {"candidate_status": "document_failed"}

# --- STUBBED: Compliance Agent (for now) ---
def run_compliance_agent(state: dict) -> dict:
    st.session_state.log.append("--- 🛡️ Running Compliance Agent (Simulation) ---")
    st.session_state.log.append(f"--- ✅ Simulated background check passed. IT provisioning initiated. ---")
    return {
        "compliance_status": "passed",
        "candidate_status": "hired"
    }

def handle_rejection(state: dict) -> dict:
    st.session_state.log.append("--- ⛔ Handling Rejection ---")
    score = state.get("screening_report", {}).get("score", "N/A")
    st.session_state.log.append(f"--- ⛔ Candidate rejected. (Score: {score}) ---")
    return {
        "candidate_status": "rejected"
    }

# --- 3. Define Graph Logic ---

class AgentState(TypedDict):
    job_description: str
    resume: str
    candidate_status: Optional[str]
    screening_report: Optional[ScreeningReport]
    interview_schedule: Optional[str]
    offer_letter_url: Optional[str]
    compliance_status: Optional[str]

def decide_after_screening(state: AgentState) -> str:
    st.session_state.log.append("--- 🚦 Deciding next step after screening ---")
    score = state.get("screening_report", {}).get("score", 0)
    if score >= 70:
        st.session_state.log.append(f"--- ✅ Score is {score}. Proceeding to Scheduling. ---")
        return "scheduling"
    else:
        st.session_state.log.append(f"--- ⛔ Score is {score}. Proceeding to Rejection. ---")
        return "rejection"

@st.cache_resource
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("screening", run_screening_agent)
    workflow.add_node("scheduling", run_real_scheduling_agent)
    workflow.add_node("document", run_real_document_agent)
    workflow.add_node("compliance", run_compliance_agent)
    workflow.add_node("rejection", handle_rejection)
    
    workflow.set_entry_point("screening")
    workflow.add_conditional_edges(
        "screening",
        decide_after_screening,
        {"scheduling": "scheduling", "rejection": "rejection"}
    )
    workflow.add_edge("scheduling", "document")
    workflow.add_edge("document", "compliance")
    workflow.add_edge("compliance", END)
    workflow.add_edge("rejection", END)
    return workflow.compile()

app = build_graph()

# --- 4. Streamlit UI ---

st.title("🤖 Agentic AI HR Recruitment System")
st.markdown("This interface allows you to test the multi-agent recruitment pipeline visually.")

if "log" not in st.session_state:
    st.session_state.log = []
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Inputs")
    
    default_jd = """Job Title: Senior Python Developer
Description: We are seeking a Senior Python Developer with 8+ years
of experience. Must be proficient in Django, PostgreSQL, and AWS
(S3, EC2, Lambda). Experience with NoSQL databases like MongoDB
and testing frameworks like Pytest is a major plus."""
    
    jd_input = st.text_area("Job Description", value=default_jd, height=200)
    
    st.divider()
    
    st.subheader("Candidate Resume")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    
    default_resume_text = """John Doe | Senior Software Engineer
Email: johndoe-test@example.com
Experience: 10 years
Summary: Senior developer skilled in Python, Django, and AWS.
Skills:
- Python, Django, Flask
- PostgreSQL, MongoDB, Redis
- AWS (EC2, S3, Lambda, SQS)
- Pytest, Unittest
- Docker, Kubernetes
"""
    resume_input_text = st.text_area("Or Paste Resume Text", value=default_resume_text, height=300)
    
    resume_content = ""
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    resume_content += page.extract_text()
                st.sidebar.success("PDF Resume Parsed!")
            elif uploaded_file.type == "text/plain":
                resume_content = uploaded_file.getvalue().decode("utf-8")
                st.sidebar.success("TXT Resume Parsed!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")
            resume_content = ""
    
    if not resume_content:
        resume_content = resume_input_text 

    if st.button("Run Recruitment Pipeline"):
        if not resume_content.strip():
            st.sidebar.error("Please upload or paste a resume.")
        else:
            st.session_state.log = []
            st.session_state.final_state = None
            st.session_state.run_complete = False
            
            inputs = {
                "job_description": jd_input,
                "resume": resume_content
            }
            
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            with st.spinner("Recruitment pipeline in progress..."):
                final_state = {}
                for event in app.stream(inputs, config=config, stream_mode="values"):
                    final_state = event 
                    
                    if "candidate_status" in final_state:
                        last_log = st.session_state.log[-1] if st.session_state.log else ""
                        new_status_log = f"--- 🛰️ Status updated to: {final_state['candidate_status']} ---"
                        if new_status_log != last_log:
                            st.session_state.log.append(new_status_log)
                
                st.session_state.final_state = final_state
                st.session_state.run_complete = True
                st.session_state.log.append("--- 🎉 Pipeline Finished ---")
            
            st.success("Pipeline finished!")

# --- Main Page for Results ---
if st.session_state.run_complete and st.session_state.final_state:
    
    state = st.session_state.final_state
    report = state.get("screening_report")

    if report:
        st.header("AI Screening Report")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Overall Score", f"{report.get('score', 0)}/100")
            st.subheader("Final Status")
            
            status = state.get('candidate_status', 'N/A').upper()
            if status == "REJECTED":
                st.error(status)
            elif status == "HIRED":
                st.success(status)
            elif "FAILED" in status:
                st.error(status)
            else:
                st.info(status)

        with col2:
            st.subheader("Screening Summary")
            st.write(report.get('summary', 'No summary provided.'))
            st.subheader("Extracted Candidate Info")
            st.text(f"Name: {report.get('first_name')} {report.get('last_name')}\nEmail: {report.get('email')}")
        
        st.divider()

        if state.get("candidate_status") != "rejected":
            st.subheader("Key Strengths")
            st.dataframe(report.get('key_strengths', []), use_container_width=True, column_config={"value": "Strength"})
            
            st.subheader("Potential Gaps & Verification Points")
            st.dataframe(report.get('potential_gaps', []), use_container_width=True, column_config={"value": "Gap"})
                
            st.subheader("AI-Suggested Interview Questions")
            st.code('\n'.join(f"{i+1}. {q}" for i, q in enumerate(report.get('suggested_interview_questions', []))), language='markdown')
            
    else:
        st.error("Screening failed to produce a report.")
    
    st.divider()

    st.header("Pipeline Agent Status")
    
    # Scheduling Agent
    if state.get("interview_schedule"):
        with st.expander("🗓️ Scheduling Agent (Calendly)", expanded=True):
            st.success("Interview Scheduled")
            st.info(f"Calendly Invitee URI: {state['interview_schedule']}")
            st.markdown("*(This invite was created via the Calendly API.)*")
    elif state.get("candidate_status") == "scheduling_failed":
         with st.expander("🗓️ Scheduling Agent (Calendly)", expanded=True):
            st.error("Scheduling Failed")
            st.markdown("*(Check the execution log and your .env file.)*")

    # Document Agent
    if state.get("offer_letter_url"):
        with st.expander("📄 Document Agent (DocuSign)", expanded=True):
            st.success("Offer Letter Sent")
            st.info(f"DocuSign {state['offer_letter_url']}")
            st.markdown("*(This envelope was created and sent via the DocuSign API.)*")
    elif state.get("candidate_status") == "document_failed":
         with st.expander("📄 Document Agent (DocuSign)", expanded=True):
            st.error("Document Sending Failed")
            st.markdown("*(Check logs, .env file, and DocuSign private key/template setup.)*")

    # Compliance Agent
    if state.get("compliance_status") == "passed":
        with st.expander("🛡️ Compliance Agent", expanded=True):
            st.success("Compliance Passed (Simulation)")
            st.info("Simulated background check cleared.")
    
    # Rejection
    if state.get("candidate_status") == "rejected":
        with st.expander("⛔ Rejection", expanded=True):
            st.error("Candidate Rejected")
            st.info(f"Reason: Low screening score ({report.get('score', 'N/A')}).")

    st.divider()
    st.subheader("Pipeline Execution Log")
    st.text_area("Log", value='\n'.join(st.session_state.log), height=300, disabled=True)

else:
    st.info("Upload or paste a resume in the sidebar and click 'Run' to start the pipeline.")