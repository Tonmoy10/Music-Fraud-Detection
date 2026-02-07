import pandas as pd
import uuid
import random
from faker import Faker
import os

# Initialize Faker
fake = Faker()

# --- CONFIGURATION ---
NUM_ROWS = 1000
LEGIT_RATIO = 0.8
FARM_RATIO = 0.1
BOT_RATIO = 0.1

# --- PATH CONFIGURATION ---
# We use a raw string (r"...") to handle Windows backslashes correctly
OUTPUT_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"
OUTPUT_FILE = "synthetic_music_fraud_data.csv"
FULL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Directory created: {path}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return False
    return True

def generate_synthetic_data():
    data = []
    
    # Shared Assets for Content Farms (Identity Hopping)
    farm_ips = [fake.ipv4() for _ in range(5)]
    farm_devices = [str(uuid.uuid4()) for _ in range(3)]
    farm_hashes = [str(uuid.uuid4()) for _ in range(10)]
    
    # Subnet for Bots
    bot_subnet = "192.168.1."
    
    # 1. LEGITIMATE DATA (80%) -> ALLOW / REVIEW
    for _ in range(int(NUM_ROWS * LEGIT_RATIO)):
        # 5% chance of being flagged for REVIEW (Gray area)
        is_review = random.random() < 0.05
        category = "REVIEW" if is_review else "ALLOW"
        
        # Profile distribution for legit users
        p_type = random.choices(
            ["normal_user", "new_user", "power_user"], 
            weights=[0.7, 0.2, 0.1], k=1
        )[0]

        row = {
            "account_external_id": str(uuid.uuid4()),
            "account_type": random.choice(["Individual", "Band", "Label"]),
            "display_name": fake.name(),
            "upload_external_id": str(uuid.uuid4()),
            "metadata_title": fake.sentence(nb_words=3).replace(".", ""),
            "metadata_genre": random.choice(["Pop", "Rock", "Jazz", "Classical", "Hip Hop", "Indie"]),
            "metadata_duration_seconds": random.randint(180, 300),
            "metadata_bitrate": random.choice([256, 320]),
            "metadata_format": "mp3",
            "metadata_collaborators": random.choice([None, fake.name()]),
            "metadata_album": fake.word().capitalize(),
            "metadata_year": random.randint(2020, 2024),
            "content_ref": f"s3://bucket/{uuid.uuid4()}",
            "fingerprints_audio_hash": str(uuid.uuid4()),
            "fingerprints_perceptual_hash": str(uuid.uuid4()),
            "device_context_device_hash": str(uuid.uuid4()),
            "device_context_ip": fake.ipv4(),
            "device_context_user_agent": fake.user_agent(),
            "expected_category": category,
            "profile_type": p_type
        }
        data.append(row)

    # 2. CONTENT FARMS (10%) -> REJECT (Identity Hoppers)
    for _ in range(int(NUM_ROWS * FARM_RATIO)):
        row = {
            "account_external_id": str(uuid.uuid4()),
            "account_type": "Label_Aggregator",
            "display_name": fake.company(),
            "upload_external_id": str(uuid.uuid4()),
            "metadata_title": fake.catch_phrase(),
            "metadata_genre": "Lo-Fi",
            "metadata_duration_seconds": random.randint(120, 240),
            "metadata_bitrate": 320,
            "metadata_format": "wav",
            "metadata_collaborators": None,
            "metadata_album": "Compilation Vol " + str(random.randint(1, 100)),
            "metadata_year": 2024,
            "content_ref": f"s3://bucket/{uuid.uuid4()}",
            "fingerprints_audio_hash": random.choice(farm_hashes), # DUPLICATE
            "fingerprints_perceptual_hash": str(uuid.uuid4()),
            "device_context_device_hash": random.choice(farm_devices), # SHARED
            "device_context_ip": random.choice(farm_ips), # SHARED
            "device_context_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "expected_category": "REJECT",
            "profile_type": "identity_hopper"
        }
        data.append(row)

    # 3. BOTS (10%) -> QUARANTINE (Spam Creators)
    for i in range(int(NUM_ROWS * BOT_RATIO)):
        row = {
            "account_external_id": str(uuid.uuid4()),
            "account_type": "Individual",
            "display_name": f"User_{random.randint(10000, 99999)}",
            "upload_external_id": str(uuid.uuid4()),
            "metadata_title": f"Track {i}",
            "metadata_genre": "Noise",
            "metadata_duration_seconds": random.randint(30, 45),
            "metadata_bitrate": 128,
            "metadata_format": "mp3",
            "metadata_collaborators": None,
            "metadata_album": None,
            "metadata_year": 2025,
            "content_ref": f"s3://bucket/{uuid.uuid4()}",
            "fingerprints_audio_hash": str(uuid.uuid4()),
            "fingerprints_perceptual_hash": str(uuid.uuid4()),
            "device_context_device_hash": str(uuid.uuid4()),
            "device_context_ip": f"{bot_subnet}{random.randint(1, 255)}",
            "device_context_user_agent": "Python-urllib/3.8",
            "expected_category": "QUARANTINE",
            "profile_type": "spam_creator"
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

if __name__ == "__main__":
    if ensure_directory_exists(OUTPUT_DIR):
        print(f"Generating data to: {FULL_PATH}")
        df_synthetic = generate_synthetic_data()
        df_synthetic.to_csv(FULL_PATH, index=False)
        print("SUCCESS: Data generated.")
        print(f"First 3 rows:\n{df_synthetic[['expected_category', 'profile_type']].head(3)}")
    else:
        print("FAILURE: Could not access E: drive path. Check your connection.")