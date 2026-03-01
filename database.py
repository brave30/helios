import os
import pymongo
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def save_call_to_db(conversation_detail):
    """
    Saves the user inputs and call metadata to MongoDB.
    """
    uri = os.getenv("MONGODB_URI")
    if not uri:
        print("⚠️  MONGODB_URI not found in environment. Skipping database save.")
        return False

    try:
        client = pymongo.MongoClient(uri)
        db = client["my-health-compass"]
        collection = db["users"]

        # Extract transcript
        transcript = conversation_detail.get("transcript", [])
        user_messages = [msg.get("message") for msg in transcript if msg.get("role") == "user"]

        # Extract metadata
        conv_id = conversation_detail.get("conversation_id")
        user_phone = conversation_detail.get("user_id") # This usually contains the phone number in Twilio calls
        
        start_time_unix = conversation_detail.get("metadata", {}).get("start_time_unix_secs")
        if start_time_unix:
            dt_object = datetime.fromtimestamp(start_time_unix)
            date_str = dt_object.strftime("%Y-%m-%d")
            time_str = dt_object.strftime("%H:%M:%S")
        else:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

        doc = {
            "conversation_id": conv_id,
            "user_id": user_phone,
            "date": date_str,
            "time": time_str,
            "user_inputs": user_messages,
            "full_transcript": transcript,
            "created_at": datetime.utcnow()
        }

        result = collection.insert_one(doc)
        print(f"✅ Transcript saved to MongoDB (ID: {result.inserted_id})")
        return True

    except Exception as e:
        print(f"❌ MongoDB Error: {e}")
        return False

if __name__ == "__main__":
    # Test connection
    print("Testing MongoDB Connection...")
    save_call_to_db({"conversation_id": "test_id", "transcript": [{"role": "user", "message": "Test message"}]})
