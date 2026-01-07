import os
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, TypedDict

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from docusign_esign import ApiClient, EnvelopesApi, EnvelopeDefinition, TemplateRole

from fastapi import UploadFile, File

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()
# ------------------------
# APP
# ------------------------
app = FastAPI(title="Agentic Recruitment Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React / Next.js
        "http://localhost:5173",   # Vite
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# LLM SETUP
# ------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.0
)

# ------------------------
# SCHEMAS
# ------------------------
class ApplicationCreate(BaseModel):
    job_id: str
    job_description: str
    resume_text: str

class ScreeningReport(TypedDict):
    score: int
    first_name: str
    last_name: str
    email: str
    key_strengths: List[str]
    potential_gaps: List[str]
    suggested_interview_questions: List[str]
    summary: str

class ScheduleRequest(BaseModel):
    scheduled_at: datetime
    timezone: str = "Asia/Calcutta"


class ShortlistDecision(BaseModel):
    decision: str  # hire | reject

# ------------------------
# IN-MEMORY STORE
# ------------------------
APPLICATIONS = {}
LOGS = {}

# ------------------------
# UTILITIES
# ------------------------
def log(app_id: str, msg: str):
    if app_id not in LOGS:
        LOGS[app_id] = []
    LOGS[app_id].append(msg)
    print(f"[{app_id}] {msg}")

# ------------------------
# 1. APPLICATION SUBMISSION
# ------------------------
@app.post("/api/applications")
def submit_application(payload: ApplicationCreate):
    app_id = str(uuid.uuid4())
    APPLICATIONS[app_id] = {
        "job_id": payload.job_id,
        "job_description": payload.job_description,
        "resume": payload.resume_text,
        "status": "submitted",
        "thread_id": str(uuid.uuid4()),
    }
    log(app_id, f"Application submitted for job {payload.job_id}")
    return {"application_id": app_id, "status": "submitted"}

# ------------------------
# 2. AI SCREENING
# ------------------------
@app.post("/api/applications/{application_id}/screen")
def run_screening(application_id: str):
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(404, "Application not found")

    prompt_template = """
    You are an expert HR screening agent. Analyze the resume vs job description.
    Extract candidate's full name and email.
    Return *only* JSON:
    {{
        "score": <int, 0-100>,
        "first_name": "<string>",
        "last_name": "<string>",
        "email": "<string>",
        "key_strengths": [<string>],
        "potential_gaps": [<string>],
        "suggested_interview_questions": [<string>],
        "summary": "<string>"
    }}

    Job Description:
    {job_description}

    Resume:
    {resume}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = JsonOutputParser(pydantic_object=ScreeningReport)
    chain = prompt | llm | parser

    try:
        report: ScreeningReport = chain.invoke({
            "job_description": app_data["job_description"],
            "resume": app_data["resume"]
        })
        app_data["screening_report"] = report

        # Apply threshold logic
        if report["score"] >= 70:
            app_data["status"] = "screening_passed"
            log(application_id, f"Screening passed with score {report['score']}")
        else:
            app_data["status"] = "rejected"
            log(application_id, f"Screening failed with score {report['score']}")

        return {"status": app_data["status"], "screening_report": report}

    except Exception as e:
        log(application_id, f"Screening failed: {e}")
        app_data["status"] = "screening_failed"
        return {"status": "screening_failed", "error": str(e)}


@app.post("/api/applications/{application_id}/upload_resume")
async def upload_resume(application_id: str, file: UploadFile = File(...)):
    """
    Upload a resume file (PDF or TXT) and store its text in the application.
    """
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(404, "Application not found")
    
    try:
        resume_text = ""
        if file.content_type == "application/pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(file.file)
            for page in reader.pages:
                resume_text += page.extract_text() or ""
        elif file.content_type == "text/plain":
            resume_text = (await file.read()).decode("utf-8")
        else:
            raise HTTPException(400, "Unsupported file type. Only PDF and TXT allowed.")

        app_data["resume"] = resume_text
        log(application_id, f"Resume uploaded and parsed ({file.filename})")
        return {"status": "resume_uploaded", "resume_text_preview": resume_text[:200]}  # first 200 chars

    except Exception as e:
        log(application_id, f"Resume upload/parsing failed: {e}")
        raise HTTPException(500, f"Failed to upload or parse resume: {e}")

# ------------------------
# 3. INTERVIEW SCHEDULING
# ------------------------
# @app.post("/api/applications/{application_id}/schedule")
# def schedule_interview(application_id: str):
#     app_data = APPLICATIONS.get(application_id)
#     if not app_data:
#         raise HTTPException(404, "Application not found")

#     if app_data["status"] != "screening_passed":
#         raise HTTPException(400, "Candidate not eligible for scheduling")

#     report = app_data.get("screening_report")
#     tz_calcutta = timezone(timedelta(hours=5, minutes=30))
#     start_time_local = datetime.now(tz_calcutta) + timedelta(days=5)
#     start_time_local = start_time_local.replace(hour=11, minute=0, second=0, microsecond=0)
#     start_time_utc = start_time_local.astimezone(timezone.utc)

#     payload = {
#         "event_type": os.getenv("CALENDLY_EVENT_TYPE_URL"),
#         "start_time": start_time_utc.isoformat(),
#         "invitee": {
#             "first_name": report["first_name"],
#             "last_name": report["last_name"],
#             "email": report["email"],
#             "timezone": "Asia/Calcutta"
#         },
#         "location": {"kind": "google_conference"}
#     }

#     headers = {
#         "Authorization": f"Bearer {os.getenv('CALENDLY_TOKEN')}",
#         "Content-Type": "application/json"
#     }

#     r = requests.post("https://api.calendly.com/invitees", headers=headers, data=json.dumps(payload))

#     if r.status_code not in (200, 201):
#         log(application_id, f"Calendly scheduling failed: {r.text}")
#         return {"status": "scheduling_failed", "error": r.text}

#     app_data["status"] = "interview_scheduled"
#     app_data["interview_schedule"] = r.json().get("resource", {}).get("uri")
#     log(application_id, f"Interview scheduled: {app_data['interview_schedule']}")
#     return {"status": app_data["status"], "interview_schedule": app_data["interview_schedule"]}

# @app.post("/api/applications/{application_id}/schedule")
# def schedule_interview(application_id: str, payload: ScheduleRequest):
#     app_data = APPLICATIONS.get(application_id)
#     if not app_data:
#         raise HTTPException(status_code=404, detail="Application not found")

#     # Must have passed screening
#     if app_data["status"] not in ["screening_passed", "shortlisted"]:
#         raise HTTPException(
#             status_code=400,
#             detail="Candidate not eligible for scheduling"
#         )

#     report = app_data.get("screening_report")
#     if not report:
#         raise HTTPException(
#             status_code=400,
#             detail="Screening report missing"
#         )

#     # Parse and normalize requested time
#     requested_time = payload.scheduled_at

#     if requested_time.tzinfo is None:
#         # Assume frontend sent UTC if timezone missing
#         requested_time = requested_time.replace(tzinfo=timezone.utc)

#     # Convert to IST (Asia/Calcutta)
#     tz_calcutta = timezone(timedelta(hours=5, minutes=30))
#     requested_time_ist = requested_time.astimezone(tz_calcutta)

#     # Enforce business hours: 9 AM – 5 PM IST
#     if not (9 <= requested_time_ist.hour < 17):
#         raise HTTPException(
#             status_code=400,
#             detail="Interview must be scheduled between 9 AM and 5 PM IST"
#         )

#     # Convert back to UTC for Calendly
#     start_time_utc = requested_time_ist.astimezone(timezone.utc)

#     payload_calendly = {
#         "event_type": os.getenv("CALENDLY_EVENT_TYPE_URL"),
#         "start_time": start_time_utc.isoformat(),
#         "invitee": {
#             "first_name": report["first_name"],
#             "last_name": report["last_name"],
#             "email": report["email"],
#             "timezone": payload.timezone or "Asia/Calcutta"
#         },
#         "location": {"kind": "google_conference"}
#     }

#     headers = {
#         "Authorization": f"Bearer {os.getenv('CALENDLY_TOKEN')}",
#         "Content-Type": "application/json"
#     }

#     response = requests.post(
#         "https://api.calendly.com/invitees",
#         headers=headers,
#         data=json.dumps(payload_calendly)
#     )

#     if response.status_code not in (200, 201):
#         log(application_id, f"Calendly scheduling failed: {response.text}")
#         raise HTTPException(
#             status_code=500,
#             detail="Calendly scheduling failed"
#         )

#     app_data["status"] = "interview_scheduled"
#     app_data["interview_schedule"] = response.json().get("resource", {}).get("uri")

#     log(
#         application_id,
#         f"Interview scheduled at {requested_time_ist.isoformat()}"
#     )

#     return {
#         "status": "interview_scheduled",
#         "interview_time_ist": requested_time_ist.isoformat(),
#         "interview_schedule": app_data["interview_schedule"]
#     }


@app.post("/api/applications/{application_id}/schedule")
def schedule_interview(application_id: str, payload: ScheduleRequest):
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(status_code=404, detail="Application not found")

    # Must have passed screening or be shortlisted
    if app_data["status"] not in ["screening_passed", "shortlisted"]:
        raise HTTPException(
            status_code=400,
            detail="Candidate not eligible for scheduling"
        )

    report = app_data.get("screening_report")
    if not report:
        raise HTTPException(
            status_code=400,
            detail="Screening report missing"
        )

    # Parse and normalize requested time
    requested_time = payload.scheduled_at
    if requested_time.tzinfo is None:
        requested_time = requested_time.replace(tzinfo=timezone.utc)

    # Convert to IST (Asia/Calcutta)
    tz_calcutta = timezone(timedelta(hours=5, minutes=30))
    requested_time_ist = requested_time.astimezone(tz_calcutta)

    # Enforce business hours: 9 AM – 5 PM IST
    if not (9 <= requested_time_ist.hour < 17):
        raise HTTPException(
            status_code=400,
            detail="Interview must be scheduled between 9 AM and 5 PM IST"
        )

    # Convert back to UTC for Calendly
    start_time_utc = requested_time_ist.astimezone(timezone.utc)

    payload_calendly = {
        "event_type": os.getenv("CALENDLY_EVENT_TYPE_URL"),
        "start_time": start_time_utc.isoformat(),
        "invitee": {
            "first_name": report["first_name"],
            "last_name": report["last_name"],
            "email": report["email"],
            "timezone": payload.timezone or "Asia/Calcutta"
        },
        "location": {"kind": "google_conference"}
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('CALENDLY_TOKEN')}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://api.calendly.com/invitees",
        headers=headers,
        data=json.dumps(payload_calendly)
    )

    if response.status_code not in (200, 201):
        log(application_id, f"Calendly scheduling failed: {response.text}")
        raise HTTPException(
            status_code=500,
            detail="Calendly scheduling failed"
        )

    # Keep the original shortlist status, track interview separately
    app_data["interview_status"] = "scheduled"
    app_data["interview_schedule"] = response.json().get("resource", {}).get("uri")

    log(
        application_id,
        f"Interview scheduled at {requested_time_ist.isoformat()}"
    )

    return {
        "status": app_data.get("status"),  # preserve original status (screening_passed / shortlisted)
        "interview_time_ist": requested_time_ist.isoformat(),
        "interview_schedule": app_data["interview_schedule"]
    }


# ------------------------
# 4. POST-INTERVIEW DECISION
# ------------------------
@app.put("/api/applications/{application_id}/shortlist")
def recruiter_decision(application_id: str, payload: ShortlistDecision):
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(404, "Application not found")

    app_data["post_interview_decision"] = payload.decision

    if payload.decision == "reject":
        app_data["status"] = "rejected"
        log(application_id, "Candidate rejected post-interview")
        return {"status": "rejected"}
    
    app_data["status"] = "shortlisted"
    log(application_id, "Candidate shortlisted post-interview")
    return {"status": "shortlisted"}

# ------------------------
# 5. DOCUSIGN OFFER
# ------------------------
@app.post("/api/applications/{application_id}/offer")
def send_offer(application_id: str):
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(404, "Application not found")

    if app_data.get("status") != "shortlisted":
        raise HTTPException(400, "Candidate not eligible for offer")

    report = app_data["screening_report"]

    # JWT Auth
    api_client = ApiClient()
    api_client.host = os.getenv("DOCUSIGN_BASE_PATH")
    # with open(os.getenv("DOCUSIGN_PRIVATE_KEY_FILE"), "r") as f:
    #     private_key = f.read()
    private_key = os.getenv("DOCUSIGN_PRIVATE_KEY")
    token = api_client.request_jwt_user_token(
        client_id=os.getenv("DOCUSIGN_INTEGRATION_KEY"),
        user_id=os.getenv("DOCUSIGN_USER_ID"),
        oauth_host_name="account-d.docusign.com",
        private_key_bytes=private_key,
        expires_in=3600,
        scopes=["signature", "impersonation"]
    )
    api_client.set_default_header("Authorization", f"Bearer {token.access_token}")

    template_role = TemplateRole(
        email=report["email"],
        name=f"{report['first_name']} {report['last_name']}",
        role_name="Candidate"
    )

    envelope_definition = EnvelopeDefinition(
        template_id=os.getenv("DOCUSIGN_TEMPLATE_ID"),
        template_roles=[template_role],
        status="sent"
    )

    envelopes_api = EnvelopesApi(api_client)
    result = envelopes_api.create_envelope(
    account_id=os.getenv("DOCUSIGN_ACCOUNT_ID"),
    envelope_definition=envelope_definition
    )


    app_data["status"] = "offer_sent"
    app_data["offer_letter_url"] = f"Envelope ID: {result.envelope_id}"
    log(application_id, f"Offer sent: {result.envelope_id}")
    return {"status": app_data["status"], "offer_letter_url": app_data["offer_letter_url"]}

# ------------------------
# 6. BACKGROUND CHECK (Simulation)
# ------------------------
@app.post("/api/applications/{application_id}/compliance")
def run_compliance(application_id: str):
    app_data = APPLICATIONS.get(application_id)
    if not app_data:
        raise HTTPException(404, "Application not found")

    if app_data.get("status") != "offer_sent":
        raise HTTPException(400, "Candidate not eligible for compliance")

    app_data["status"] = "hired"
    app_data["compliance_status"] = "passed"
    log(application_id, "Compliance check passed, candidate hired")
    return {"status": app_data["status"], "compliance_status": "passed"}

# ------------------------
# 7. GET LOGS
# ------------------------
@app.get("/api/applications/{application_id}/logs")
def get_logs(application_id: str):
    return {"logs": LOGS.get(application_id, [])}

# For Render / uvicorn compatibility
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
