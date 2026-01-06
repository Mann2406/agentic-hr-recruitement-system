import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Optional

# --- 1. Define the LLM for the Screening Agent ---
# We use Groq as you suggested for high speed.
# Using Llama 3 70b for strong reasoning.
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.0
)

# --- 2. Define the Screening Agent ---

class ScreeningReport(TypedDict):
    """The JSON structure for the screening report."""
    score: int
    key_strengths: list[str]
    potential_gaps: list[str]
    suggested_interview_questions: list[str]
    summary: str

def run_screening_agent(state: dict):
    """
    Runs the LLM-powered screening agent.
    
    Takes the job_description and resume from the state,
    generates an explainable report, and updates the state.
    """
    print("--- 🧠 Running Screening Agent ---")
    
    job_description = state["job_description"]
    resume = state["resume"]

    # This is the core of your agent. The prompt is everything.
    # It instructs the LLM to act as an expert recruiter
    # and to return *only* a JSON object.
    prompt_template = """
    You are an expert HR screening agent. Your task is to analyze the
    provided resume against the job description.

    Return *only* a JSON object that matches the following structure:
    {{
        "score": <int, 0-100 score of relevance>,
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
    
    # Define the output parser to enforce JSON
    parser = JsonOutputParser(pydantic_object=ScreeningReport)

    # Create the agentic chain
    chain = prompt | llm | parser

    # Invoke the chain
    try:
        report = chain.invoke({
            "job_description": job_description,
            "resume": resume
        })
        
        print(f"--- ✅ Screening Report Generated (Score: {report['score']}) ---")
        
        # This dictionary is the new state
        return {
            "screening_report": report,
            "candidate_status": "screening_passed"
        }
    except Exception as e:
        print(f"--- ❌ Error in Screening Agent ---")
        print(e)
        return {
            "candidate_status": "screening_failed",
            "screening_report": {"summary": f"Failed to parse resume: {e}", "score": 0}
        }

# --- 3. Stubbed Agents (For you to implement) ---

def run_scheduling_agent(state: dict):
    """
    STUB: This is where you would integrate with Google/Outlook Calendar APIs.
    """
    print("--- 🗓️ Running Scheduling Agent ---")
    # TODO: Add API calls to Google Calendar
    # 1. Get interviewers' availability
    # 2. Find common slots
    # 3. Email candidate (or use RAG assistant)
    # 4. Book the meeting
    
    candidate_name = state.get("resume", "Candidate")[:15] # Placeholder
    
    # For now, we just simulate success.
    simulated_schedule = "2025-11-05 at 10:00 AM"
    print(f"--- ✅ Simulated interview for {candidate_name} on {simulated_schedule} ---")
    
    return {
        "interview_schedule": simulated_schedule,
        "candidate_status": "interview_scheduled"
    }

def run_document_agent(state: dict):
    """
    STUB: This is where you would integrate with Jinja2, WeasyPrint, and DocuSign.
    """
    print("--- 📄 Running Document Agent ---")
    # TODO: Add logic for document generation
    # 1. Pull data (name, salary, start_date) from state
    # 2. Use Jinja2 to populate an HTML offer letter template
    # 3. Use WeasyPrint to convert HTML -> PDF
    # 4. Make API call to DocuSign to send for signature
    
    simulated_url = "https://docusign.com/offer-letter-xyz123"
    print(f"--- ✅ Simulated offer letter sent via DocuSign ---")
    
    return {
        "offer_letter_url": simulated_url,
        "candidate_status": "offer_sent"
    }

def run_compliance_agent(state: dict):
    """
    STUB: This is where you would integrate with Checkr and Admin SDKs.
    """
    print("--- 🛡️ Running Compliance Agent ---")
    # TODO: Add API calls for compliance
    # 1. Make API call to Checkr for background check
    # 2. Wait for Checkr webhook
    # 3. Make API calls to Google Admin SDK / Microsoft Graph to create user
    
    print(f"--- ✅ Simulated background check passed. IT provisioning initiated. ---")
    
    return {
        "compliance_status": "passed",
        "candidate_status": "hired"
    }

def handle_rejection(state: dict):
    """
    A final node to handle candidates who are rejected.
    """
    print("--- ⛔ Handling Rejection ---")
    # TODO: Add logic to send a polite rejection email
    
    report = state.get("screening_report", {})
    score = report.get("score", "N/A")
    summary = report.get("summary", "Not a fit at this time.")
    
    print(f"--- ⛔ Candidate rejected. Score: {score}. Reason: {summary} ---")
    
    return {
        "candidate_status": "rejected"
    }