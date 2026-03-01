import urllib.parse
import os
import json
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from groq import Groq
import pymongo
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_SEARCH = "https://rarediseases.org/"

FIRST_RESULT_XPATH = "(//div[contains(@class,'results-grouping')]//article//h5//a)[1]"
SYMPTOMS_TAB_XPATH  = "//div[contains(@class,'reports-anchor-nav')]//li[@data-section='symptoms']"
SYMPTOMS_BLOCK_XPATH = "(//*[@id='symptoms'] | //*[@data-section='symptoms' and (self::section or self::div)])[1]"

def get_symptoms_text(driver, disease: str) -> dict:
    import time
    
    # 1) go to search
    q = urllib.parse.quote_plus(disease)
    search_url = (
        f"{BASE_SEARCH}?s={q}"
        f"&rdb-search=true"
        f"&post_type%5B%5D=rare-diseases"
        f"&post_type%5B%5D=gard-rare-disease"
    )
    driver.get(search_url)
    
    # Wait for Cloudflare challenge if present
    time.sleep(3)

    wait = WebDriverWait(driver, 15)

    # 2) first result link
    first_link_el = wait.until(EC.presence_of_element_located((By.XPATH, FIRST_RESULT_XPATH)))
    first_url = first_link_el.get_attribute("href")

    # 3) open disease page
    driver.get(first_url)
    
    # Wait for Cloudflare challenge on disease page
    time.sleep(3)

    # 4) click "Signs & Symptoms" tab (some pages may not require click, but safe)
    tab = wait.until(EC.element_to_be_clickable((By.XPATH, SYMPTOMS_TAB_XPATH)))
    tab.click()

    # 5) extract symptoms block text
    block = wait.until(EC.presence_of_element_located((By.XPATH, SYMPTOMS_BLOCK_XPATH)))
    symptoms_text = block.text.strip()

    return {
        "disease": disease,
        "page_url": first_url,
        "symptoms": symptoms_text
    }


def generate_tracking_questions(symptoms_text: str) -> list[dict]:
    """
    Use Groq API to generate 3 symptom tracking questions with tags from symptoms text.
    Returns list of dicts with 'question' and 'tag' keys.
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""Based on the following disease symptoms, generate exactly 3 simple questions 
for daily symptom tracking. Each question should be answerable with a number on a scale of 1-10 
or a simple yes/no.

Symptoms:
{symptoms_text}

For each question, also provide a short tag (1-3 words) that categorizes what the question measures.

Requirements:
- Questions should be simple and easy to answer quickly
- Focus on the most important/common symptoms
- Include the scale or response type in each question (e.g., "on a scale of 1-10" or "yes/no")
- Tags should be short category labels like: "Pain Level", "Breathing", "Mobility", "Fatigue", "Sleep Quality", "Mood", etc.
- Return ONLY a JSON array of objects, no other text

Example output:
[{{"question": "How would you rate your pain level today on a scale of 1-10?", "tag": "Pain Level"}}, {{"question": "Did you experience any difficulty breathing today? (yes/no)", "tag": "Breathing"}}, {{"question": "Rate your energy level on a scale of 1-10", "tag": "Energy"}}]
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a medical assistant that generates symptom tracking questions. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Parse JSON response
    try:
        # Try to extract JSON array from response
        if response_text.startswith("["):
            items = json.loads(response_text)
        else:
            # Try to find JSON array in response
            import re
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                items = json.loads(match.group())
            else:
                raise ValueError("No JSON array found in response")
        
        # Normalize response - ensure each item has question and tag
        result = []
        for item in items[:3]:
            if isinstance(item, dict):
                result.append({
                    "question": item.get("question", ""),
                    "tag": item.get("tag", "General")
                })
            elif isinstance(item, str):
                # Fallback for old format
                result.append({"question": item, "tag": "General"})
        
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse Groq response: {response_text}")
        raise e


def save_questions_to_mongodb(user_id: str, questions: list[dict]) -> bool:
    """
    Save the generated questions to the user's metrics in MongoDB.
    Each question has 'question' and 'tag' keys.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI not configured")
    
    client = pymongo.MongoClient(uri)
    db = client["my-health-compass"]
    users = db["users"]
    
    from bson import ObjectId
    
    # Build metrics array with questions and tags
    metrics = []
    for item in questions:
        question = item.get("question", "") if isinstance(item, dict) else item
        tag = item.get("tag", "General") if isinstance(item, dict) else "General"
        
        # Determine metric type based on question content
        if "yes/no" in question.lower() or "yes or no" in question.lower():
            metric_type = "boolean"
        else:
            metric_type = "scale"
        
        metrics.append({
            "name": question,
            "tag": tag,
            "type": metric_type,
            "value": None  # Will be filled by the AI calling service
        })
    
    # Update user document
    result = users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"metrics": metrics}}
    )
    
    client.close()
    
    return result.modified_count > 0


def process_disease_for_user(driver, disease: str, user_id: str) -> dict:
    """
    Full pipeline: scrape symptoms -> generate questions -> save to MongoDB.
    """
    # 1. Scrape symptoms
    print(f"🔍 Scraping symptoms for '{disease}'...")
    symptoms_data = get_symptoms_text(driver, disease)
    
    # 2. Generate questions using Groq
    print("🤖 Generating tracking questions with Groq...")
    questions = generate_tracking_questions(symptoms_data["symptoms"])
    
    print(f"✅ Generated {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. [{q['tag']}] {q['question']}")
    
    # 3. Save to MongoDB
    print(f"💾 Saving questions to MongoDB for user {user_id}...")
    saved = save_questions_to_mongodb(user_id, questions)
    
    if saved:
        print("✅ Questions saved successfully!")
    else:
        print("⚠️ User document not found or no changes made")
    
    return {
        "disease": symptoms_data["disease"],
        "page_url": symptoms_data["page_url"],
        "questions": questions,
        "saved_to_mongodb": saved
    }