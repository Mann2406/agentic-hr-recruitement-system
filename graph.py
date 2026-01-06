import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import the agent functions from our other file
from agents import (
    run_screening_agent,
    run_scheduling_agent,
    run_document_agent,
    run_compliance_agent,
    handle_rejection,
    ScreeningReport
)

# --- 1. Define the Agentic State ---
# This TypedDict is the "memory" or "state" of our multi-agent system.
# It gets passed to each agent (node) in the graph.
class AgentState(TypedDict):
    job_description: str
    resume: str
    candidate_status: Optional[str]
    screening_report: Optional[ScreeningReport]
    interview_schedule: Optional[str]
    offer_letter_url: Optional[str]
    compliance_status: Optional[str]

# --- 2. Define the Conditional Routing ---
# This function decides where to go *after* the screening agent.
def decide_after_screening(state: AgentState) -> str:
    """
    Checks the screening score and decides the next step.
    
    If score is 70+, proceed to scheduling.
    Otherwise, proceed to rejection.
    """
    print("--- 🚦 Deciding next step after screening ---")
    
    score = state.get("screening_report", {}).get("score", 0)
    
    if score >= 70:
        print(f"--- ✅ Score is {score}. Proceeding to Scheduling. ---")
        return "scheduling"
    else:
        print(f"--- ⛔ Score is {score}. Proceeding to Rejection. ---")
        return "rejection"

# --- 3. Build the Graph ---
def build_graph():
    """
    Creates the LangGraph workflow.
    """
    workflow = StateGraph(AgentState)

    # --- Add Nodes ---
    # Each node is one of our agents.
    workflow.add_node("screening", run_screening_agent)
    workflow.add_node("scheduling", run_scheduling_agent)
    workflow.add_node("document", run_document_agent)
    workflow.add_node("compliance", run_compliance_agent)
    workflow.add_node("rejection", handle_rejection)

    # --- Add Edges ---
    
    # The workflow starts at the "screening" node
    workflow.set_entry_point("screening")
    
    # --- This is the critical conditional edge ---
    # After "screening", call `decide_after_screening` to choose the next node.
    workflow.add_conditional_edges(
        "screening",
        decide_after_screening,
        {
            "scheduling": "scheduling", # If function returns "scheduling", go to that node
            "rejection": "rejection"    # If function returns "rejection", go to that node
        }
    )
    
    # --- Add the rest of the linear flow ---
    workflow.add_edge("scheduling", "document")
    workflow.add_edge("document", "compliance")
    
    # --- Define the end points of the graph ---
    workflow.add_edge("compliance", END)
    workflow.add_edge("rejection", END)
    
    # Compile the graph into a runnable application
    return workflow.compile()

# --- 4. Run the Application ---
if __name__ == "__main__":
    
    # Load the .env file (for GROQ_API_KEY)
    load_dotenv()
    
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found. Please set it in your .env file.")
        exit(1)

    app = build_graph()

    # --- Define our inputs ---
    job_description = """
    Job Title: Senior Python Developer
    Description: We are seeking a Senior Python Developer with 8+ years
    of experience. Must be proficient in Django, PostgreSQL, and AWS
    (S3, EC2, Lambda). Experience with NoSQL databases like MongoDB
    and testing frameworks like Pytest is a major plus.
    """

    # --- SCENARIO 1: A GOOD CANDIDATE ---
    print("=" * 40)
    print("--- 🚀 RUNNING SCENARIO 1: GOOD CANDIDATE ---")
    print("=" * 40)
    
    resume_good = """
    John Doe | Senior Software Engineer
    
    Experience: 10 years
    Summary: Senior developer skilled in Python, Django, and AWS.
    
    Skills:
    - Python, Django, Flask
    - PostgreSQL, MongoDB, Redis
    - AWS (EC2, S3, Lambda, SQS)
    - Pytest, Unittest
    - Docker, Kubernetes
    """
    
    inputs = {
        "job_description": job_description,
        "resume": resume_good
    }
    
    # A unique thread_id for this run
    config = {"configurable": {"thread_id": "run-1-good"}}
    
    # 'stream_mode="values"' shows the full state after each step
    for event in app.stream(inputs, config=config, stream_mode="values"):
        print("\n--- 💾 CURRENT STATE ---")
        # Print the keys of the node that just ran
        print(f"Node: {list(event.keys())[0]}")
        # Print the full state
        print(event)
        print("-" * 20)

    # --- SCENARIO 2: A BAD CANDIDATE ---
    print("\n" * 3)
    print("=" * 40)
    print("--- 🚀 RUNNING SCENARIO 2: BAD CANDIDATE ---")
    print("=" * 40)
    
    resume_bad = """
    Jane Smith | Junior Web Developer
    
    Experience: 1 year
    Summary: Eager to learn!
    
    Skills:
    - HTML, CSS, JavaScript
    - React
    - Node.js
    """
    
    inputs_bad = {
        "job_description": job_description,
        "resume": resume_bad
    }
    
    config_bad = {"configurable": {"thread_id": "run-2-bad"}}
    
    for event in app.stream(inputs_bad, config=config_bad, stream_mode="values"):
        print("\n--- 💾 CURRENT STATE ---")
        print(f"Node: {list(event.keys())[0]}")
        print(event)
        print("-" * 20)