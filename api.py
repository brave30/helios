"""
api.py - Health Compass AI Calling API
======================================
Main endpoints:
- POST /generate-questions: Generate symptom tracking questions from disease name
- POST /make-call: Trigger AI call, collect responses, save to MongoDB
- GET /user/{user_id}: Get user data
- GET /health: Health check
"""

import os
import re
import time
import json
import requests
from datetime import datetime, timezone
from typing import List, Optional, Any
from bson import ObjectId
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pymongo
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health Compass AI Calling API",
    description="AI-powered symptom tracking via phone calls",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# File-based cache for disease questions
CACHE_FILE = os.path.join(os.path.dirname(__file__), "disease_cache.json")

def load_cache() -> dict:
    """Load cache from JSON file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache: dict):
    """Save cache to JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

_questions_cache: dict = load_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise HTTPException(status_code=500, detail="MONGODB_URI not configured")
    client = pymongo.MongoClient(uri)
    return client["my-health-compass"]


def convert_objectids(obj):
    """Recursively convert ObjectIds to strings for JSON serialization."""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids(item) for item in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Request Models
# ─────────────────────────────────────────────────────────────────────────────

class GenerateQuestionsRequest(BaseModel):
    user_id: str
    disease_name: str


class MakeCallRequest(BaseModel):
    user_id: str


class FlareRoutineRequest(BaseModel):
    user_id: str
    flare_threshold: float = 7.0  # Average metric value above this triggers flare


class FlareAlertCallRequest(BaseModel):
    user_id: str
    phone_number: Optional[str] = None  # Optional override, defaults to MY_PHONE_NUMBER


# ─────────────────────────────────────────────────────────────────────────────
# Transcript Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_metric_value(answer: str, metric_type: str) -> Any:
    """Parse user's answer based on metric type (scale/boolean)."""
    answer_lower = answer.lower().strip()
    
    if metric_type == "scale":
        # Extract first number
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return int(numbers[0])
        # Word to number mapping
        word_map = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'once': 1, 'twice': 2, 'thrice': 3,
            'never': 0, 'rarely': 1, 'sometimes': 3, 'often': 5,
            'frequently': 7, 'always': 10, 'daily': 7,
            'yes': 1, 'yeah': 1, 'no': 0, 'nope': 0
        }
        for word, num in word_map.items():
            if word in answer_lower:
                return num
        return None
    
    elif metric_type == "boolean":
        yes_words = ['yes', 'yeah', 'yep', 'true', 'i do', 'i did', 'i have']
        no_words = ['no', 'nope', 'false', "don't", "didn't", "haven't", 'not']
        for word in yes_words:
            if word in answer_lower:
                return True
        for word in no_words:
            if word in answer_lower:
                return False
        return None
    
    return answer


def extract_key_terms(text: str) -> set:
    """Extract key terms from text for matching."""
    import string
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    stop_words = {'your', 'child', 'does', 'has', 'have', 'been', 'is', 'the', 'a', 'an',
                  'how', 'often', 'do', 'you', 'and', 'to', 'for', 'of', 'in', 'on', 'my'}
    return set(text.split()) - stop_words


def match_metric_to_question(agent_message: str, metric_name: str) -> bool:
    """Check if agent message is asking about a metric."""
    agent_terms = extract_key_terms(agent_message)
    metric_terms = extract_key_terms(metric_name)
    if not metric_terms:
        return False
    overlap = agent_terms & metric_terms
    return len(overlap) / len(metric_terms) >= 0.5


def extract_metrics_from_transcript(transcript: list, user_metrics: list) -> list:
    """Extract metric values from transcript based on user's configured metrics."""
    extracted = []
    remaining = list(user_metrics)
    
    for i, msg in enumerate(transcript):
        if msg.get("role", "").lower() == "agent" and remaining:
            for metric in remaining:
                if match_metric_to_question(msg.get("message", ""), metric["name"]):
                    if i + 1 < len(transcript) and transcript[i + 1].get("role", "").lower() == "user":
                        value = parse_metric_value(transcript[i + 1].get("message", ""), metric.get("type", "scale"))
                        if value is not None:
                            extracted.append({
                                "name": metric["name"],
                                "value": value,
                                "metricType": metric.get("type", "scale")
                            })
                            remaining.remove(metric)
                            break
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# ElevenLabs Integration
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(questions: list) -> str:
    """Build system prompt with questions for the AI agent."""
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""You are a warm, friendly personal check-in assistant named SecondSense.

Your ONLY job is to ask these questions, in order, one at a time:

{numbered}

Rules:
- Greet briefly, then ask question 1.
- After each answer, acknowledge briefly (1 sentence), then next question.
- Do NOT skip or reorder questions.
- Stay on topic. Redirect if needed: "I'll note that — let's continue with the questions."
- After the last question, thank the user and say goodbye.
- Keep responses SHORT (2-3 sentences max).
"""


def update_agent(agent_id: str, api_key: str, questions: list) -> bool:
    """Update ElevenLabs agent with user-specific questions."""
    response = requests.patch(
        f"{ELEVENLABS_BASE_URL}/convai/agents/{agent_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "name": "SecondSense Caller",
            "conversation_config": {
                "agent": {
                    "prompt": {"prompt": build_system_prompt(questions), "llm": "gemini-2.0-flash", "temperature": 0.5},
                    "first_message": "Hey! This is your SecondSense check-in call. I have a few quick questions — just answer out loud. Ready? Let's begin.",
                    "language": "en",
                },
                "tts": {"voice_id": "EXAVITQu4vr4xnSDxMaL"},
            },
        },
    )
    return response.status_code in (200, 201)


def update_agent_for_flare_alert(agent_id: str, api_key: str, child_name: str, condition: str) -> bool:
    """Update ElevenLabs agent for flare alert notification call."""
    prompt = f"""You are a caring health assistant named SecondSense making an important notification call.

Your task is to:
1. Inform them that {child_name} has entered a FLARE state for their {condition}
2. Advise them to check the SecondSense app for the updated flare care routine
3. Ask if they have any immediate questions
4. Remind them to reach out to their healthcare provider if symptoms worsen
5. End the call with reassurance

Rules:
- Be calm, compassionate, and supportive
- Keep the call brief but informative (under 2 minutes)
- Do not provide medical advice beyond what's in the app
- If they ask detailed questions, direct them to their healthcare provider
"""
    
    first_message = f"Hi, this is SecondSense calling with an important health update about {child_name}. Do you have a moment?"
    
    response = requests.patch(
        f"{ELEVENLABS_BASE_URL}/convai/agents/{agent_id}",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "name": "SecondSense Flare Alert",
            "conversation_config": {
                "agent": {
                    "prompt": {"prompt": prompt, "llm": "gemini-2.0-flash", "temperature": 0.5},
                    "first_message": first_message,
                    "language": "en",
                },
                "tts": {"voice_id": "EXAVITQu4vr4xnSDxMaL"},
            },
        },
    )
    return response.status_code in (200, 201)


def trigger_call(agent_id: str, phone_number_id: str, to_number: str, api_key: str) -> dict:
    """Trigger outbound call via ElevenLabs/Twilio."""
    response = requests.post(
        f"{ELEVENLABS_BASE_URL}/convai/twilio/outbound-call",
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"agent_id": agent_id, "agent_phone_number_id": phone_number_id, "to_number": to_number},
    )
    if response.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Call failed: {response.text}")
    return response.json()


def wait_for_conversation(agent_id: str, api_key: str, existing_ids: set, max_wait: int = 300) -> dict:
    """Poll until a new completed conversation appears."""
    headers = {"xi-api-key": api_key}
    new_conv_id = None
    elapsed = 0
    
    print(f"🔍 Waiting for new conversation (ignoring {len(existing_ids)} existing)...")
    
    while elapsed < max_wait:
        time.sleep(5)
        elapsed += 5
        
        try:
            response = requests.get(
                f"{ELEVENLABS_BASE_URL}/convai/conversations",
                headers=headers,
                params={"agent_id": agent_id, "page_size": 10},
            )
            if response.status_code != 200:
                continue
            
            for conv in response.json().get("conversations", []):
                conv_id = conv.get("conversation_id")
                if conv_id in existing_ids:
                    continue
                
                if new_conv_id is None:
                    new_conv_id = conv_id
                    print(f"📞 Found: {conv_id} ({conv.get('status')})")
                
                if conv_id == new_conv_id:
                    if conv.get("status") == "done":
                        print(f"✅ Completed: {conv_id}")
                        detail = requests.get(f"{ELEVENLABS_BASE_URL}/convai/conversations/{conv_id}", headers=headers)
                        if detail.status_code == 200:
                            return detail.json()
                    else:
                        print(f"⏳ {conv_id} still {conv.get('status')}... ({elapsed}s)")
                        break
        except Exception as e:
            print(f"⚠️ Error: {e}")
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/generate-questions")
async def generate_questions(request: GenerateQuestionsRequest):
    """
    Generate symptom tracking questions from a disease name.
    
    Pipeline:
    1. Scrape symptoms from rarediseases.org
    2. Generate 3 tracking questions using Groq AI
    3. Save questions to user's metrics in MongoDB
    """
    import undetected_chromedriver as uc
    from scrapper import get_symptoms_text, generate_tracking_questions, save_questions_to_mongodb
    
    user_id = request.user_id
    disease_name = request.disease_name.strip()
    
    if not disease_name:
        raise HTTPException(status_code=400, detail="disease_name is required")
    
    # Verify user exists
    db = get_db()
    try:
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid user_id: {e}")
    
    # Check cache first
    cache_key = disease_name.lower()
    if cache_key in _questions_cache:
        print(f"📦 Cache hit for '{disease_name}'")
        cached = _questions_cache[cache_key]
        # Still save to MongoDB for this user
        saved = save_questions_to_mongodb(user_id, cached["questions"])
        return {
            "success": True,
            "disease": cached["disease"],
            "disease_url": cached["disease_url"],
            "questions": cached["questions"],
            "saved": saved,
            "cached": True
        }
    
    # Setup headless Chrome
    options = uc.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    driver = None
    try:
        driver = uc.Chrome(options=options, version_main=145, headless=True)
        driver.implicitly_wait(5)
        
        print(f"🔍 Scraping symptoms for '{disease_name}'...")
        symptoms_data = get_symptoms_text(driver, disease_name)
        
        print("🤖 Generating questions with Groq...")
        questions = generate_tracking_questions(symptoms_data["symptoms"])
        
        print(f"💾 Saving to MongoDB...")
        saved = save_questions_to_mongodb(user_id, questions)
        
        # Cache the result
        _questions_cache[cache_key] = {
            "disease": symptoms_data["disease"],
            "disease_url": symptoms_data["page_url"],
            "questions": questions
        }
        save_cache(_questions_cache)
        print(f"📦 Cached questions for '{disease_name}'")
        
        return {
            "success": True,
            "disease": symptoms_data["disease"],
            "disease_url": symptoms_data["page_url"],
            "questions": questions,
            "saved": saved,
            "cached": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if driver:
            driver.quit()


@app.post("/make-call")
async def make_call(request: MakeCallRequest):
    """
    Trigger AI call, wait for completion, extract metrics, save to MongoDB.
    
    Pipeline:
    1. Fetch user's questions from metrics
    2. Update ElevenLabs agent
    3. Make outbound call
    4. Wait for completion
    5. Extract answers from transcript
    6. Save to logs
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    phone_id = os.getenv("ELEVENLABS_PHONE_NUMBER_ID")
    to_number = os.getenv("MY_PHONE_NUMBER")
    
    if not all([api_key, agent_id, phone_id, to_number]):
        raise HTTPException(status_code=500, detail="Missing environment variables")
    
    db = get_db()
    collection = db["users"]
    
    try:
        user_oid = ObjectId(request.user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    user = collection.find_one({"_id": user_oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    metrics = user.get("metrics", [])
    if not metrics:
        raise HTTPException(status_code=400, detail="User has no metrics configured")
    
    questions = [m["name"] for m in metrics[:3] if m.get("name")]
    if not questions:
        raise HTTPException(status_code=400, detail="No valid questions found")
    
    # Update agent
    if not update_agent(agent_id, api_key, questions):
        raise HTTPException(status_code=500, detail="Failed to update agent")
    
    # Get existing conversations
    existing_ids = set()
    try:
        resp = requests.get(
            f"{ELEVENLABS_BASE_URL}/convai/conversations",
            headers={"xi-api-key": api_key},
            params={"agent_id": agent_id, "page_size": 20},
        )
        if resp.status_code == 200:
            existing_ids = {c["conversation_id"] for c in resp.json().get("conversations", [])}
    except:
        pass
    
    # Make call
    call_result = trigger_call(agent_id, phone_id, to_number, api_key)
    print(f"📞 Call initiated: {call_result}")
    
    # Wait for completion
    conversation = wait_for_conversation(agent_id, api_key, existing_ids)
    
    if not conversation:
        return {"success": False, "message": "Call timed out", "questions": questions}
    
    transcript = conversation.get("transcript", [])
    if not transcript:
        return {"success": False, "message": "No transcript", "conversation_id": conversation.get("conversation_id")}
    
    # Extract metrics
    extracted = extract_metrics_from_transcript(transcript, metrics)
    
    if not extracted:
        return {
            "success": False,
            "message": "Could not extract metrics",
            "conversation_id": conversation.get("conversation_id"),
            "transcript": transcript
        }
    
    # Save to MongoDB
    now = datetime.now(timezone.utc)
    new_log = {
        "_id": ObjectId(),
        "time": now,
        "metrics": [{"_id": ObjectId(), **m} for m in extracted]
    }
    
    collection.update_one(
        {"_id": user_oid},
        {"$push": {"logs": new_log}, "$set": {"lastCheckinTime": now, "updatedAt": now}}
    )
    
    return {
        "success": True,
        "message": f"Saved {len(extracted)} metrics",
        "conversation_id": conversation.get("conversation_id"),
        "questions": questions,
        "extracted_metrics": extracted,
        "log_id": str(new_log["_id"])
    }


@app.get("/user/{user_id}")
async def get_user(user_id: str):
    """Get user data by ID."""
    db = get_db()
    try:
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return convert_objectids(user)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/generate-flare-routine")
async def generate_flare_routine(request: FlareRoutineRequest):
    """
    Generate a modified routine for flare weeks.
    
    Pipeline:
    1. Fetch user's routine data and recent symptom logs
    2. Analyze symptoms to detect flare week
    3. Send to Groq to generate adjusted routine
    4. Save flare tasks/meds to MongoDB
    """
    from groq import Groq
    
    db = get_db()
    collection = db["users"]
    
    try:
        user_oid = ObjectId(request.user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    user = collection.find_one({"_id": user_oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user data
    condition = user.get("condition", "Unknown condition")
    child_name = user.get("childName", "the patient")
    routine_tasks = user.get("routineTasks", [])
    medications = user.get("medications", [])
    logs = user.get("logs", [])
    
    # Analyze recent symptoms (last 7 days)
    recent_logs = logs[-7:] if len(logs) >= 7 else logs
    
    if not recent_logs:
        raise HTTPException(status_code=400, detail="No symptom logs available")
    
    # Calculate symptom severity
    all_values = []
    symptom_summary = []
    
    for log in recent_logs:
        log_time = log.get("time", "Unknown time")
        for metric in log.get("metrics", []):
            name = metric.get("name", "")
            value = metric.get("value")
            metric_type = metric.get("metricType", "scale")
            
            if metric_type == "scale" and isinstance(value, (int, float)):
                all_values.append(value)
                symptom_summary.append(f"- {name}: {value}/10")
            elif metric_type == "boolean" and value is not None:
                symptom_summary.append(f"- {name}: {'Yes' if value else 'No'}")
                if value:  # True = symptom present
                    all_values.append(8)  # Treat as high severity
    
    if not all_values:
        raise HTTPException(status_code=400, detail="No measurable symptoms found")
    
    avg_severity = sum(all_values) / len(all_values)
    is_flare = avg_severity >= request.flare_threshold
    
    # Format routine data for prompt
    routine_str = "\n".join([f"- {t.get('name', t)} (category: {t.get('category', 'general')}, time: {t.get('time', 'N/A')})" for t in routine_tasks]) if routine_tasks else "No routine tasks defined"
    meds_str = "\n".join([f"- {m.get('name', m)} ({m.get('dose', 'N/A')}, {m.get('time', 'N/A')})" for m in medications]) if medications else "No medications defined"
    symptoms_str = "\n".join(symptom_summary)
    
    # Build Groq prompt
    prompt = f"""You are a medical care specialist helping manage a patient with a rare disease during a flare period.

Patient: {child_name}
Condition: {condition}
Current Status: {'FLARE WEEK - symptoms are elevated and require adjusted care' if is_flare else 'Normal week - monitoring'}
Average Symptom Severity: {avg_severity:.1f}/10

Current Daily Routine:
{routine_str}

Current Medications:
{meds_str}

Recent Symptom Log (past check-ins):
{symptoms_str}

Based on the elevated symptoms, generate a flare week care plan by MODIFYING the existing routine tasks.

CRITICAL RULE - ONE-TO-ONE TRANSFORMATION:
- You MUST generate EXACTLY the same number of flareTasks as there are routine tasks in "Current Daily Routine"
- Each routine task should have ONE corresponding flare task
- Do NOT add extra tasks that don't exist in the routine
- If routine has 1 task → generate 1 flare task
- If routine has 3 tasks → generate 3 flare tasks

HOW TO TRANSFORM EACH ROUTINE TASK:
For each task in the routine, create a flare-adjusted version with:
- Reduced duration, frequency, or intensity
- Added breaks, rest periods, or modifications
- Specific accommodations for the current symptoms
- Keep the same time slot as the original task

EXAMPLES OF HOW TO TRANSFORM:
- "Homework" at 6 PM → "Limit homework to 15-20 min with 10 min rest breaks" at 6 PM ✓
- "Soccer Practice" at 4 PM → "Skip soccer, do 10 min gentle stretching instead" at 4 PM ✓
- "School" at 8 AM → "Request reduced workload from teacher for 3 days" at 8 AM ✓
- "Bath Time" at 7 PM → "Warm bath with Epsom salts for 15 min" at 7 PM ✓
- "Playtime" at 3 PM → "Quiet indoor activities only, no running" at 3 PM ✓

CRITICAL - DO NOT:
- Add "gentle" or "light" in front of the original name (e.g., "Gentle Homework" ❌)
- Generate more tasks than exist in the routine
- Create tasks unrelated to the current routine

CATEGORY VALUES (use exactly one):
- "medications" → pill/medicine tasks
- "care" → physical comfort, symptom relief
- "nutrition" → food, drinks, diet
- "school" → school/activity adjustments
- "admin" → tracking, appointments, communication

TIME VALUES: Keep the same time as the original routine task, or use "All Day", "Morning", "Afternoon", "Evening", "Night"

Respond with JSON:
{{
  "isFlare": true,
  "severity": "mild" | "moderate" | "severe",
  "alertLevel": "green" | "yellow" | "red",
  "flareTasks": [
    {{"name": "SPECIFIC ACTION with details", "category": "medications|care|nutrition|school|admin", "time": "same as original routine task"}}
  ],
  "recommendations": ["specific actionable recommendation 1", "specific actionable recommendation 2", "specific actionable recommendation 3"],
  "message": "Brief supportive message for the caregiver",
  "tip": "A detailed tip (2-3 sentences) specific to {condition} flare management with warning signs and when to seek help."
}}

IMPORTANT: Generate EXACTLY {len(routine_tasks)} flare task(s) - one for each routine task.
Respond ONLY with valid JSON."""
    
    # Call Groq API
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medical assistant that generates care routines for rare disease patients. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        flare_data = json.loads(result_text)
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Groq response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")
    
    # Save flare tasks to MongoDB if it's a flare week
    if flare_data.get("isFlare", is_flare):
        flare_tasks = flare_data.get("flareTasks", [])
        
        # Ensure each task has proper schema: {name, category, time, _id}
        for task in flare_tasks:
            task["_id"] = ObjectId()
            # Ensure required fields exist with proper defaults
            if "category" not in task:
                task["category"] = "care"
            if "time" not in task:
                task["time"] = "All Day"
        
        collection.update_one(
            {"_id": user_oid},
            {
                "$set": {
                    "flareTasks": flare_tasks,
                    "isFlareEnabled": True,
                    "mode": "flare",
                    "updatedAt": datetime.now(timezone.utc)
                }
            }
        )
    
    return {
        "success": True,
        "avgSeverity": round(avg_severity, 2),
        "isFlare": flare_data.get("isFlare", is_flare),
        "severity": flare_data.get("severity", "unknown"),
        "alertLevel": flare_data.get("alertLevel", "yellow"),
        "flareTasks": convert_objectids(flare_data.get("flareTasks", [])),
        "recommendations": flare_data.get("recommendations", []),
        "message": flare_data.get("message", ""),
        "tip": flare_data.get("tip", "")
    }


@app.post("/flare-alert-call")
async def flare_alert_call(request: FlareAlertCallRequest):
    """
    Make a phone call to notify caregiver that patient has entered flare state.
    
    Pipeline:
    1. Fetch user data from MongoDB
    2. Update ElevenLabs agent with flare alert message
    3. Trigger outbound call
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    phone_id = os.getenv("ELEVENLABS_PHONE_NUMBER_ID")
    to_number = request.phone_number or os.getenv("MY_PHONE_NUMBER")
    
    if not all([api_key, agent_id, phone_id, to_number]):
        raise HTTPException(status_code=500, detail="Missing environment variables")
    
    db = get_db()
    collection = db["users"]
    
    try:
        user_oid = ObjectId(request.user_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    user = collection.find_one({"_id": user_oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user info
    child_name = user.get("childName", "your child")
    condition = user.get("condition", "their condition")
    caregiver_name = user.get("caregiverName", "Caregiver")
    
    # Update agent for flare alert
    if not update_agent_for_flare_alert(agent_id, api_key, child_name, condition):
        raise HTTPException(status_code=500, detail="Failed to update agent")
    
    # Trigger the call
    try:
        call_result = trigger_call(agent_id, phone_id, to_number, api_key)
        print(f"📞 Flare alert call initiated to {to_number}: {call_result}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Call failed: {e}")
    
    # Update user mode to flare
    collection.update_one(
        {"_id": user_oid},
        {
            "$set": {
                "isFlareEnabled": True,
                "mode": "flare",
                "updatedAt": datetime.now(timezone.utc)
            }
        }
    )
    
    return {
        "success": True,
        "message": f"Flare alert call initiated to {to_number}",
        "childName": child_name,
        "condition": condition,
        "callResult": call_result
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
