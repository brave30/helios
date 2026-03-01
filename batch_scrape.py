"""
batch_scrape.py - Batch scrape and cache disease questions
============================================================
Run this script to pre-populate the cache with questions for rare diseases.
"""

import os
import json
import time
import undetected_chromedriver as uc
from dotenv import load_dotenv

load_dotenv()

# Import scraping functions
from scrapper import get_symptoms_text, generate_tracking_questions

CACHE_FILE = os.path.join(os.path.dirname(__file__), "disease_cache.json")

rare_diseases = [
    "Recurrent Respiratory Papillomatosis (RRP)",
    "Alagille Syndrome",
    "Batten Disease",
    "Ehlers-Danlos Syndrome",
    "Dravet Syndrome",
    "Bardet-Biedl Syndrome",
    "Cystinosis",
    "Angelman Syndrome",
    "Chiari Malformation",
    "Alpha-1 Antitrypsin Deficiency",
    "CHARGE Syndrome",
    "Costello Syndrome",
    "Canavan Disease",
    "Bloom Syndrome",
    "Erdheim-Chester Disease",
    "Aicardi Syndrome",
    "Becker Muscular Dystrophy",
    "Cornelia de Lange Syndrome",
    "Dermatomyositis (rare autoimmune form)",
    "Cat Eye Syndrome",
    "Autoimmune Encephalitis",
    "Chediak-Higashi Syndrome",
    "Adult-Onset Still Disease",
    "Aspartylglucosaminuria",
    "Congenital Myasthenic Syndrome",
    "Ellis-van Creveld Syndrome",
    "Ataxia Telangiectasia",
    "Brugada Syndrome",
    "Cohen Syndrome",
    "Acrodermatitis Enteropathica",
    "DiGeorge Syndrome",
    "Blue Rubber Bleb Nevus Syndrome",
    "Citrullinemia",
    "Alexander Disease",
    "Bartter Syndrome",
    "Apert Syndrome",
    "Dandy-Walker Malformation",
    "Amyloidosis (Primary)",
    "Camurati-Engelmann Disease",
    "Duchenne Muscular Dystrophy",
    "Arteriovenous Malformation",
    "Emery-Dreifuss Muscular Dystrophy",
    "Degos Disease",
    "Aniridia",
    "Central Core Disease",
    "Allan-Herndon-Dudley Syndrome",
    "Aase Syndrome",
    "Coffin-Lowry Syndrome",
    "Bohring-Opitz Syndrome",
    "Alport Syndrome",
    "Achondrogenesis",
    "Charcot-Marie-Tooth Disease",
    "Aarskog Syndrome",
    "Crouzon Syndrome",
    "Antiphospholipid Syndrome",
    "Diamond-Blackfan Anemia",
    "Congenital Central Hypoventilation Syndrome",
    "Beta-Mannosidosis",
    "Behcet Disease",
    "Cleidocranial Dysplasia",
    "Birt-Hogg-Dube Syndrome",
    "Addison Disease (Primary Adrenal Insufficiency)",
    "Acromegaly",
    "Carpenter Syndrome",
    "Autoimmune Hepatitis",
    "Alkaptonuria",
    "Acute Intermittent Porphyria",
    "Arthrogryposis Multiplex Congenita",
    "Cri du Chat Syndrome",
    "Adrenoleukodystrophy",
    "Cronkhite-Canada Syndrome",
    "Achondroplasia",
    "Cytomegalovirus Infection (congenital severe form)",
    "Celiac Artery Compression Syndrome",
    "Budd-Chiari Syndrome",
    "Epidermolysis Bullosa",
    "Arginosuccinic Aciduria",
    "Castleman Disease",
    "Dyggve-Melchior-Clausen Syndrome",
    "Aplastic Anemia",
    "Acid Maltase Deficiency",
    "Albers-Schonberg Disease",
    "Bowen-Conradi Syndrome",
    "Alternating Hemiplegia of Childhood",
    "CHARGE Syndrome (variant forms)",
    "Adams-Oliver Syndrome",
    "Canavan Disease (infantile form)",
    "Alpha-Mannosidosis",
    "Carcinoid Syndrome",
    "Atrial Septal Defect (genetic rare forms)",
    "Asherman Syndrome",
    "Chiari Malformation Type II",
    "Central Core Myopathy",
    "Erdheim-Chester Disease (cardiac involvement)",
    "Congenital Fiber-Type Disproportion",
    "Familial Mediterranean Fever",
    "Gaucher Disease",
    "Hereditary Angioedema",
    "Hypophosphatasia",
    "Kabuki Syndrome"
]


def load_cache() -> dict:
    """Load existing cache from file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(cache: dict):
    """Save cache to file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def setup_driver():
    """Setup headless Chrome driver."""
    options = uc.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
    
    driver = uc.Chrome(options=options, version_main=145, headless=True)
    driver.implicitly_wait(5)
    return driver


def main():
    cache = load_cache()
    driver = None
    
    print(f"🚀 Starting batch scrape for {len(rare_diseases)} diseases")
    print(f"📦 Cache has {len(cache)} entries")
    
    # Count how many we need to process
    to_process = [d for d in rare_diseases if d.lower() not in cache]
    print(f"📝 Need to process: {len(to_process)} diseases")
    
    if not to_process:
        print("✅ All diseases already cached!")
        return
    
    try:
        driver = setup_driver()
        
        for i, disease in enumerate(to_process):
            cache_key = disease.lower()
            
            print(f"\n[{i+1}/{len(to_process)}] Processing: {disease}")
            
            try:
                # Scrape symptoms
                print(f"  🔍 Scraping symptoms...")
                symptoms_data = get_symptoms_text(driver, disease)
                
                # Generate questions
                print(f"  🤖 Generating questions...")
                questions = generate_tracking_questions(symptoms_data["symptoms"])
                
                # Cache result
                cache[cache_key] = {
                    "disease": symptoms_data["disease"],
                    "disease_url": symptoms_data["page_url"],
                    "questions": questions
                }
                
                # Save after each successful scrape
                save_cache(cache)
                print(f"  ✅ Cached: {len(questions)} questions")
                
                # Rate limiting - wait between requests
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                # Continue with next disease
                continue
    
    finally:
        if driver:
            driver.quit()
    
    print(f"\n🎉 Done! Cache now has {len(cache)} entries")


if __name__ == "__main__":
    main()
