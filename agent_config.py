"""
agent_config.py
---------------
Define the questions and behaviour of your ElevenLabs calling agent here.
Edit QUESTIONS and AGENT_NAME to suit your needs.
"""

import os
import pymongo
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------
AGENT_NAME = "SecondSense Caller"

# ---------------------------------------------------------------------------
# Default questions (used as fallback if no user metrics found)
# ---------------------------------------------------------------------------
DEFAULT_QUESTIONS = [
    "How are you feeling today, on a scale from one to ten?",
    "What is your top priority for today?",
    "Is there anything blocking your progress or causing you stress?",
]

# ---------------------------------------------------------------------------
# First message spoken as soon as the call connects
# ---------------------------------------------------------------------------
FIRST_MESSAGE = (
    "Hey! This is your SecondSense check-in call. "
    "I have a few quick questions for you — just answer out loud and I'll listen. Ready? Let's begin."
)


def get_user_metrics(user_id: str) -> list:
    """Fetch metrics.name from MongoDB for a given user_id."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return DEFAULT_QUESTIONS
    
    try:
        client = pymongo.MongoClient(uri)
        db = client["my-health-compass"]
        collection = db["users"]
        
        user = collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return DEFAULT_QUESTIONS
        
        metrics = user.get("metrics", [])
        if not metrics:
            return DEFAULT_QUESTIONS
        
        # Extract metric names as questions (take up to 3)
        questions = [m["name"] for m in metrics[:3] if m.get("name")]
        return questions if questions else DEFAULT_QUESTIONS
    except Exception as e:
        print(f"⚠️  Error fetching user metrics: {e}")
        return DEFAULT_QUESTIONS


# ---------------------------------------------------------------------------
# System prompt — tells the LLM how to behave
# ---------------------------------------------------------------------------
def build_system_prompt(questions: list = None) -> str:
    """Build the system prompt with the given questions or defaults."""
    if questions is None:
        questions = DEFAULT_QUESTIONS
    
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are a warm, friendly, and concise personal check-in assistant named SecondSense.

Your ONLY job on this call is to ask the user the following questions, in order, one at a time:

{numbered}

Rules:
- Greet the user briefly, then move straight into question 1.
- After the user answers, acknowledge their response with a short, encouraging remark (1 sentence max), then ask the next question.
- Do NOT skip questions or change their order.
- Do NOT go off-topic. If the user asks you something unrelated, gently redirect: "I'll note that down — let's keep going with the questions."
- After the last question is answered, thank the user warmly and say goodbye.
- Keep ALL your responses SHORT — no more than 2–3 sentences at a time.
- Speak naturally, as if you're a supportive friend making a quick check-in call.
"""
