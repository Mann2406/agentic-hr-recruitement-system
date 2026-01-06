import os
import uuid
import streamlit as st
import requests
import json
from docusign_esign import ApiClient, EnvelopesApi, EnvelopeDefinition, TemplateRole
from datetime import datetime, timedelta, timezone
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

# --- FIXED: Using a valid Groq model ---
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

def run_real_scheduling_agent(state: dict) -> dict:
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
        
        tz_calcutta = timezone(timedelta(hours=3, minutes=30))
        # Use 5 days from now to ensure we get a new slot
        target_day = datetime.now(tz_calcutta) + timedelta(days=5)
        start_time_local = target_day.replace(hour=11, minute=0, second=0, microsecond=0) # 2 PM
        start_time_utc = start_time_local.astimezone(timezone.utc)

        payload = {
            "event_type": event_type_url,
            "start_time": start_time_utc.isoformat(), 
            "invitee": {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "timezone": "Asia/Calcutta" 
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
            st.session_state.log.append(f"--- ❌ Calendly Error: {response.text} ---")
            return {"candidate_status": "scheduling_failed"}

    except Exception as e:
        st.session_state.log.append(f"--- ❌ Scheduling Agent Error: {e} ---")
        return {"candidate_status": "scheduling_failed"}

def run_real_document_agent(state: dict) -> dict:
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
            role_name="Candidate"
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

def run_compliance_agent(state: dict) -> dict:
    st.session_state.log.append("--- 🛡️ Running Compliance Agent (Simulation) ---")
    st.session_state.log.append(f"--- ✅ Simulated background check passed. IT provisioning initiated. ---")
    # This is the final step
    return {
        "compliance_status": "passed",
        "candidate_status": "hired" 
    }

def handle_rejection(state: dict) -> dict:
    """Handles candidate rejection based on the pipeline stage."""
    if state.get("post_interview_decision") == "reject":
        st.session_state.log.append("--- ⛔ Handling Post-Interview Rejection ---")
        st.session_state.log.append(f"--- ⛔ Candidate rejected by recruiter after interview. ---")
    else:
        st.session_state.log.append("--- ⛔ Handling Screening Rejection ---")
        score = state.get("screening_report", {}).get("score", "N/A")
        st.session_state.log.append(f"--- ⛔ Candidate rejected based on low score: {score} ---")
    
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
    post_interview_decision: Optional[str]


# --- Graph 1: Pre-Interview ---
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
def build_pre_interview_graph():
    """Builds the graph for Screening -> Scheduling."""
    workflow = StateGraph(AgentState)
    workflow.add_node("screening", run_screening_agent)
    workflow.add_node("scheduling", run_real_scheduling_agent)
    workflow.add_node("rejection", handle_rejection)
    
    workflow.set_entry_point("screening")
    workflow.add_conditional_edges(
        "screening",
        decide_after_screening,
        {"scheduling": "scheduling", "rejection": "rejection"}
    )
    workflow.add_edge("scheduling", END)
    workflow.add_edge("rejection", END)
    return workflow.compile()

# --- NEW: Graph 2: Post-Interview ---

# --- THIS IS THE FIX (Part 1) ---
# This is a new "dummy" node. It's the entry point for the graph
# and just returns an empty dict. It does nothing.
def post_interview_entry_point(state: AgentState) -> dict:
    """
    Dummy entry point for the post-interview graph.
    The recruiter's decision is already in the state.
    """
    st.session_state.log.append("--- 🚦 Recruiter decision received ---")
    return {} # Return an empty dict to update state

# This function is now *only* used for the conditional edge
def decide_post_interview(state: AgentState) -> str:
    """Handles the recruiter's decision after the interview."""
    if state.get("post_interview_decision") == "hire":
        st.session_state.log.append("--- ✅ Recruiter marked as 'Hire'. Proceeding to Documents. ---")
        return "document"
    else:
        st.session_state.log.append("--- ⛔ Recruiter marked as 'Reject'. ---")
        return "rejection"

@st.cache_resource
def build_post_interview_graph():
    """Builds the graph for Document -> Compliance OR Rejection."""
    workflow = StateGraph(AgentState)
    
    # --- THIS IS THE FIX (Part 2) ---
    
    # 1. Add the new dummy entry node
    workflow.add_node("post_interview_entry", post_interview_entry_point) 
    
    # 2. Add the real action nodes
    workflow.add_node("document", run_real_document_agent)
    workflow.add_node("compliance", run_compliance_agent)
    workflow.add_node("rejection", handle_rejection)
    
    # 3. Set the dummy node as the entry point
    workflow.set_entry_point("post_interview_entry")
    
    # 4. Add the conditional edge *from* the dummy node
    workflow.add_conditional_edges(
        "post_interview_entry",    # <-- Start from the entry node
        decide_post_interview,     # <-- Call this function to decide
        {                          # <-- Map the string result to the next node
            "document": "document",
            "rejection": "rejection"
        }
    )
    # --- END FIX ---
    
    # Linear flow for hiring
    workflow.add_edge("document", "compliance")
    workflow.add_edge("compliance", END)
    workflow.add_edge("rejection", END)
    return workflow.compile()

# --- 4. Streamlit UI ---

st.title("🤖 Agentic AI HR Recruitment System")
st.markdown("This interface allows you to test the multi-agent recruitment pipeline visually.")

# Initialize session state
if "log" not in st.session_state:
    st.session_state.log = []
if "final_state" not in st.session_state:
    st.session_state.final_state = {}
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

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
    
    # --- This code was missing in your last file ---
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
            # Reset state for a new run
            st.session_state.log = []
            st.session_state.final_state = {}
            st.session_state.run_complete = False
            st.session_state.thread_id = str(uuid.uuid4()) # New candidate, new thread
            
            app_pre_interview = build_pre_interview_graph()
            
            inputs = {
                "job_description": jd_input,
                "resume": resume_content
            }
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.spinner("Running pre-interview pipeline..."):
                final_state = {}
                for event in app_pre_interview.stream(inputs, config=config, stream_mode="values"):
                    final_state = event 
                    
                    if "candidate_status" in final_state:
                        last_log = st.session_state.log[-1] if st.session_state.log else ""
                        new_status_log = f"--- 🛰️ Status updated to: {final_state['candidate_status']} ---"
                        if new_status_log != last_log:
                            st.session_state.log.append(new_status_log)
                
                st.session_state.final_state = final_state
                st.session_state.run_complete = True
                st.session_state.log.append("--- 🎉 Pre-Interview Pipeline Finished ---")
            
            st.success("Pre-interview pipeline finished!")
            st.rerun() # Rerun to show the new UI state

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

    # --- Section 2: Full Pipeline Status ---
    st.header("Pipeline Agent Status")
    
    # Scheduling Agent
    if state.get("interview_schedule"):
        with st.expander("🗓️ Scheduling Agent (Calendly)", expanded=True):
            st.success("Interview Scheduled")
            st.info(f"Calendly Invitee URI: {state['interview_schedule']}")
    elif state.get("candidate_status") == "scheduling_failed":
         with st.expander("🗓️ Scheduling Agent (Calendly)", expanded=True):
            st.error("Scheduling Failed")

    # Document Agent
    if state.get("offer_letter_url"):
        with st.expander("📄 Document Agent (DocuSign)", expanded=True):
            st.success("Offer Letter Sent")
            st.info(f"DocuSign {state['offer_letter_url']}")
    elif state.get("candidate_status") == "document_failed":
         with st.expander("📄 Document Agent (DocuSign)", expanded=True):
            st.error("Document Sending Failed")

    # Compliance Agent
    if state.get("compliance_status") == "passed":
        with st.expander("🛡️ Compliance Agent", expanded=True):
            st.success("Compliance Passed (Simulation)")
            st.info("Simulated background check cleared.")
    
    # Rejection
    if state.get("candidate_status") == "rejected":
        with st.expander("⛔ Rejection", expanded=True):
            st.error("Candidate Rejected")
            reason = "Low screening score" if not state.get("post_interview_decision") else "Recruiter decision"
            st.info(f"Reason: {reason}")
    
    st.divider()
    
    # --- Human-in-the-Loop Button Logic ---
    if state.get("candidate_status") == "interview_scheduled":
        st.header("Human-in-the-Loop: Post-Interview Decision")
        st.warning("The pipeline is paused. After the interview, mark the candidate's status below.")
        
        col1, col2 = st.columns(2)
        
        post_interview_decision = None
        if col1.button("✅ Proceed to Hire"):
            post_interview_decision = "hire"
        
        if col2.button("⛔ Reject Candidate"):
            post_interview_decision = "reject"
            
        # If a button was clicked, run the second graph
        if post_interview_decision:
            with st.spinner(f"Running {post_interview_decision} pipeline..."):
                app_post = build_post_interview_graph()
                
                # Prepare the state for the second graph
                post_interview_inputs = st.session_state.final_state.copy()
                post_interview_inputs["post_interview_decision"] = post_interview_decision
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                final_state = {}
                for event in app_post.stream(post_interview_inputs, config=config, stream_mode="values"):
                    final_state = event
                    
                    if "candidate_status" in final_state:
                        last_log = st.session_state.log[-1] if st.session_state.log else ""
                        new_status_log = f"--- 🛰️ Status updated to: {final_state['candidate_status']} ---"
                        if new_status_log != last_log:
                            st.session_state.log.append(new_status_log)

                # Merge the new state into the old state
                st.session_state.final_state.update(final_state)
                st.session_state.log.append("--- 🎉 Post-Interview Pipeline Finished ---")
            
            st.success("Post-interview pipeline finished!")
            st.rerun() # Rerun the script to show the final UI

    st.subheader("Pipeline Execution Log")
    st.text_area("Log", value='\n'.join(st.session_state.log), height=300, disabled=True)

elif not st.session_state.run_complete:
    st.info("Upload or paste a resume in the sidebar and click 'Run' to start the pipeline.")