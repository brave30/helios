# Helios — AI-Powered Rare Disease Symptom Tracking

An AI-powered health companion that helps caregivers track symptoms of rare diseases through automated phone calls, generates personalized care routines during flare weeks, and provides intelligent symptom analysis.

---

## Features

- **Symptom Question Generation** — Scrapes rare disease symptoms and generates personalized tracking questions using Groq AI
- **AI Phone Calls** — Automated check-in calls via ElevenLabs voice AI + Twilio
- **Flare Detection** — Analyzes symptom logs to detect flare weeks
- **Flare Routine Generation** — AI-generated care routines adjusted for flare periods
- **Flare Alert Calls** — Automated phone notifications when patient enters flare state
- **Disease Cache** — Pre-cached questions for 100+ rare diseases

---

## Prerequisites

| Service | What you need | Where to get it |
|---|---|---|
| **ElevenLabs** | API Key + Agent ID + Phone Number ID | [elevenlabs.io](https://elevenlabs.io) |
| **Twilio** | Account SID, Auth Token, Phone Number | [twilio.com](https://twilio.com) |
| **Groq** | API Key (starts with `gsk_`) | [console.groq.com](https://console.groq.com) |
| **MongoDB** | Connection URI | [mongodb.com](https://mongodb.com) |

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

Create a `.env` file with:

```env
GROQ_API_KEY=gsk_...

# ElevenLabs
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_AGENT_ID=agent_...
ELEVENLABS_PHONE_NUMBER_ID=phnum_...

# Twilio
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...

# Target
MY_PHONE_NUMBER=+1...

# MongoDB
MONGODB_URI=mongodb+srv://...
```

### 3. Run the API

```bash
python api.py
```

Server runs at `http://localhost:8001`

---

## API Endpoints

### Health Check
```bash
GET /health
```

### Get User
```bash
GET /user/{user_id}
```

### Generate Symptom Questions
Scrapes rare disease info and generates 3 tracking questions.
```bash
POST /generate-questions
Content-Type: application/json

{
  "user_id": "mongo_user_id",
  "disease_name": "marfan syndrome"
}
```

### Make Check-in Call
Calls user, asks symptom questions, extracts answers, saves to MongoDB.
```bash
POST /make-call
Content-Type: application/json

{
  "user_id": "mongo_user_id"
}
```

### Generate Flare Routine
Analyzes symptoms and generates AI care routine for flare week.
```bash
POST /generate-flare-routine
Content-Type: application/json

{
  "user_id": "mongo_user_id",
  "flare_threshold": 5.0
}
```

Response:
```json
{
  "success": true,
  "avgSeverity": 7.1,
  "isFlare": true,
  "severity": "moderate",
  "alertLevel": "yellow",
  "flareTasks": [
    {"name": "Increased Rest", "category": "rest", "time": "All Day", "_id": "..."},
    {"name": "Pain Management", "category": "medication", "time": "Morning", "_id": "..."}
  ],
  "recommendations": ["..."],
  "message": "...",
  "tip": "Detailed medical tip for PDF..."
}
```

### Flare Alert Call
Calls caregiver to notify patient has entered flare state.
```bash
POST /flare-alert-call
Content-Type: application/json

{
  "user_id": "mongo_user_id",
  "phone_number": "+1234567890"  // optional, defaults to MY_PHONE_NUMBER
}
```

---

## Batch Scraping

Pre-populate the disease question cache for 100 rare diseases:

```bash
python batch_scrape.py
```

This saves results to `disease_cache.json` for instant responses.

---

## Project Structure

```
helios/
├── api.py              # Main FastAPI server
├── scrapper.py         # Disease symptom scraping + Groq question generation
├── batch_scrape.py     # Batch cache 100 rare diseases
├── disease_cache.json  # Cached disease questions
├── agent_config.py     # ElevenLabs agent configuration
├── create_agent.py     # One-time agent setup
├── make_call.py        # Standalone call script
├── requirements.txt    # Python dependencies
├── .env                # Credentials (never commit!)
└── README.md
```

---

## MongoDB User Schema

```json
{
  "_id": "ObjectId",
  "childName": "string",
  "condition": "string",
  "caregiverName": "string",
  "mode": "normal" | "flare",
  "isFlareEnabled": boolean,
  "metrics": [{ "name": "...", "tag": "...", "type": "scale|boolean", "value": null }],
  "routineTasks": [{ "name": "...", "category": "...", "time": "...", "_id": "..." }],
  "flareTasks": [{ "name": "...", "category": "...", "time": "...", "_id": "..." }],
  "medications": [...],
  "flareMeds": [...],
  "logs": [{ "time": "datetime", "metrics": [...] }]
}
```

---

## Categories for Tasks

`care`, `rest`, `medication`, `nutrition`, `activity`, `monitoring`, `school`, `therapy`

---

## License

MIT
