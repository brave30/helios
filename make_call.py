"""
make_call.py
------------
Triggers an outbound call from your Twilio number to your cell phone.
The ElevenLabs agent will answer the call and ask you the configured questions.

Prerequisites:
  1. Your .env must have all six variables set (see .env.example).
  2. Your Twilio trial account must have your cell number verified.
  3. Run create_agent.py first to get your ELEVENLABS_AGENT_ID.

Usage:
    python make_call.py --user 69a38f826b19f94dd08f1c44
    python make_call.py --user 69a38f826b19f94dd08f1c44 --to +1XXXXXXXXXX
"""

import os
import sys
import argparse
import requests
from dotenv import load_dotenv
from agent_config import AGENT_NAME, FIRST_MESSAGE, build_system_prompt, get_user_metrics

load_dotenv()

BASE_URL = "https://api.elevenlabs.io/v1"


def get_env(key: str) -> str:
    value = os.environ.get(key)
    if not value or value.startswith("your_"):
        print(f"❌ Missing or placeholder value for {key} in .env")
        sys.exit(1)
    return value


def update_agent_with_questions(agent_id: str, questions: list) -> None:
    """Update the ElevenLabs agent with user-specific questions."""
    api_key = get_env("ELEVENLABS_API_KEY")
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    
    payload = {
        "name": AGENT_NAME,
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": build_system_prompt(questions),
                    "llm": "gemini-2.0-flash",
                    "temperature": 0.5,
                },
                "first_message": FIRST_MESSAGE,
                "language": "en",
            },
            "tts": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",
            },
        },
    }
    
    print(f"📝 Updating agent with user's questions...")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
    
    response = requests.patch(
        f"{BASE_URL}/convai/agents/{agent_id}",
        headers=headers,
        json=payload,
    )
    
    if response.status_code not in (200, 201):
        print(f"⚠️  Warning: Could not update agent: {response.text}")
    else:
        print(f"✅ Agent updated with {len(questions)} questions from MongoDB")


import time
from get_transcript import list_conversations, fetch_and_print_transcript

def wait_for_call_completion(agent_id: str, old_latest_conv: str = None):
    print("\n⏳ Waiting for the call to finish to generate transcript...")
    print("   (You can safely Ctrl+C if you don't want to wait)")
    
    target_conv_id = None
    
    while True:
        try:
            convs = list_conversations(agent_id, page_size=5)
            if not convs:
                time.sleep(2)
                continue
                
            latest = convs[0]
            
            # Step 1: Identify the new conversation ID if we haven't yet
            if target_conv_id is None:
                if latest["conversation_id"] != old_latest_conv:
                    target_conv_id = latest["conversation_id"]
                    print(f"📡 Tracking call session: {target_conv_id} (Status: {latest['status']})")
                else:
                    # Still waiting for the new call to show up in the history list
                    time.sleep(2)
                    continue

            # Step 2: Once we have the ID, wait for it to become 'done'
            # We fetch the specific detail to be most accurate
            if target_conv_id:
                # Find the target in the list to check status
                target_in_list = next((c for c in convs if c["conversation_id"] == target_conv_id), None)
                
                if target_in_list and target_in_list["status"] == "done":
                    print(f"✨ Call finished!")
                    # Brief pause to allow summary generation/finalization on ElevenLabs side
                    time.sleep(1) 
                    fetch_and_print_transcript(target_conv_id)
                    break
            
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n👋 Stopped waiting for transcript.")
            break
        except Exception as e:
            # print(f"\n⚠️  Polling update: {e}") # Keep it quiet
            time.sleep(2)


def make_outbound_call(to_number: str, user_id: str = None) -> None:
    api_key              = get_env("ELEVENLABS_API_KEY")
    agent_id             = get_env("ELEVENLABS_AGENT_ID")
    phone_number_id      = get_env("ELEVENLABS_PHONE_NUMBER_ID")

    # Update agent with user-specific questions from MongoDB
    if user_id:
        questions = get_user_metrics(user_id)
        update_agent_with_questions(agent_id, questions)
    
    # Capture the latest conv ID before we start the call
    try:
        current_convs = list_conversations(agent_id, page_size=1)
        prev_latest_id = current_convs[0]["conversation_id"] if current_convs else None
    except:
        prev_latest_id = None

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "agent_id":              agent_id,
        "agent_phone_number_id": phone_number_id,
        "to_number":             to_number,
    }

    print(f"📞 Initiating call to {to_number}...")
    
    response = requests.post(
        f"{BASE_URL}/convai/twilio/outbound-call",
        headers=headers,
        json=payload,
    )

    if response.status_code not in (200, 201):
        print(f"\n❌ Error {response.status_code}:")
        print(response.text)
        response.raise_for_status()

    data = response.json()
    print(f"\n✅ Call placed successfully!")
    print(f"   Call SID : {data.get('call_sid', 'N/A')}")
    print(f"\n📱 Your phone should ring shortly. Answer and speak naturally!")
    
    # Wait for completion and transcript
    wait_for_call_completion(agent_id, prev_latest_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger an ElevenLabs outbound call.")
    parser.add_argument(
        "--to",
        default=None,
        help="Phone number to call. Defaults to MY_PHONE_NUMBER from .env.",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="MongoDB user _id to fetch metrics/questions from.",
    )
    args = parser.parse_args()

    target = args.to or get_env("MY_PHONE_NUMBER")
    make_outbound_call(target, user_id=args.user)
