#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Music Streaming Fraud Detection (fixed)

Generates 1000 rows of synthetic data and saves to:
    chatgpt_synthetic_music_fraud.csv

Dependencies:
    pip install pandas numpy faker
"""
import csv
import uuid
import random
import hashlib
from faker import Faker
import pandas as pd
import numpy as np

# -------------------------
# Configuration / Seed
# -------------------------
SEED = 42
NUM_ROWS = 1000
RNG = np.random.default_rng(SEED)
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# -------------------------
# Domain choices
# -------------------------
ACCOUNT_TYPES = ["Basic", "Premium", "Artist"]
GENRES = ["Pop", "Hip-Hop", "Ambient", "Noise", "Electronic", "Rock", "Classical", "Jazz"]
BITRATES = [64, 96, 128, 192, 256, 320]
FORMATS = ["mp3", "wav", "flac", "aac", "ogg"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "Spotify/8.7.0 (Linux; Android 11)",
    "curl/7.68.0",
    "python-requests/2.28.1",
]

# -------------------------
# Fraud design decisions
# -------------------------
FRAUD_RATIO = 0.22  # ~22% fraudulent accounts (bot farms, hacked)
LEGIT_RATIO = 1 - FRAUD_RATIO

# -------------------------
# Helpers
# -------------------------
def hex_hash(prefix: str, length: int = 40) -> str:
    base = (prefix + str(uuid.uuid4())).encode("utf-8")
    return hashlib.sha1(base).hexdigest()[:length]

def short_hash(length: int = 16) -> str:
    return uuid.uuid4().hex[:length]

def random_title(fake_obj: Faker, fraud: bool) -> str:
    if fraud:
        patterns = [
            "Untitled Track", "New Release", "Track " + str(RNG.integers(1, 5000)),
            "Loop " + str(RNG.integers(1, 999)), "Sample " + str(RNG.integers(1, 9999))
        ]
        title = random.choice(patterns)
        if RNG.random() < 0.25:
            title = title + " " + random.choice(["(Remix)", "(v2)", "feat. AI"])
        return title
    else:
        words = fake_obj.sentence(nb_words=3).rstrip(".")
        if RNG.random() < 0.15:
            words += " feat. " + fake_obj.name().split()[0]
        return words

def random_display_name(fake_obj: Faker, fraud: bool) -> str:
    if fraud:
        templates = [
            "Artist" + str(RNG.integers(1000, 9999)),
            "User_" + str(RNG.integers(10000, 99999)),
            "Studio" + str(RNG.integers(100, 999)),
            fake_obj.company().split()[0] + " Records"
        ]
        return random.choice(templates)
    else:
        if RNG.random() < 0.6:
            return fake_obj.name()
        else:
            return fake_obj.user_name().title()

def generate_collaborators(fake_obj: Faker, fraud: bool) -> str:
    if fraud:
        if RNG.random() < 0.75:
            return ""
        else:
            return "Various"
    else:
        if RNG.random() < 0.3:
            n = RNG.integers(1, 4)
            names = [fake_obj.name() for _ in range(n)]
            return ";".join(names)
        return ""

def generate_album(fake_obj: Faker, fraud: bool) -> str:
    if fraud:
        if RNG.random() < 0.85:
            return ""
        return "Singles"
    else:
        if RNG.random() < 0.5:
            return fake_obj.word().title() + " " + random.choice(["EP", "LP", "Collection"])
        return ""

def generate_year(fraud: bool) -> int:
    if fraud:
        return int(RNG.choice([2022, 2023, 2024, 2025], p=[0.05, 0.25, 0.45, 0.25]))
    else:
        return int(RNG.integers(2000, 2026))

def generate_bitrate(fraud: bool) -> int:
    if fraud:
        return int(RNG.choice([64, 96, 128], p=[0.5, 0.35, 0.15]))
    else:
        return int(RNG.choice([128, 192, 256, 320], p=[0.25, 0.2, 0.3, 0.25]))

def generate_duration(fraud: bool) -> int:
    if fraud:
        if RNG.random() < 0.85:
            return int(RNG.integers(20, 95))
        else:
            return int(RNG.integers(100, 240))
    else:
        return int(RNG.integers(180, 301))

def generate_genre(fraud: bool) -> str:
    if fraud:
        return RNG.choice(["Electronic", "Ambient", "Noise", "Pop"])
    else:
        return RNG.choice(GENRES)

def generate_format(fraud: bool) -> str:
    if fraud:
        return RNG.choice(["mp3", "aac", "ogg"])
    else:
        return RNG.choice(FORMATS)

def generate_account_type(fraud: bool) -> str:
    if fraud:
        return RNG.choice(["Artist", "Basic"])
    else:
        return RNG.choice(ACCOUNT_TYPES)

def generate_profile_type(fraud: bool) -> str:
    if fraud:
        return RNG.choice(["bot_farm", "hacked_account"], p=[0.85, 0.15])
    else:
        return RNG.choice(["normal_user", "indie_artist", "label_artist"], p=[0.6, 0.3, 0.1])

def generate_ip(fraud: bool) -> str:
    if fraud and RNG.random() < 0.6:
        prefix = random.choice(["45.77.12.", "185.62.12.", "103.21.58."])
        return prefix + str(RNG.integers(2, 250))
    else:
        return fake.ipv4_public()

def generate_device_hash(fraud: bool) -> str:
    if fraud and RNG.random() < 0.5:
        return "dev-" + short_hash(12)
    else:
        return "dev-" + short_hash(16)

def generate_user_agent(fraud: bool) -> str:
    if fraud:
        if RNG.random() < 0.5:
            return random.choice(["curl/7.68.0", "python-requests/2.28.1", "Bot/1.0"])
        else:
            return random.choice(USER_AGENTS)
    else:
        return random.choice(USER_AGENTS)

# -------------------------
# Row generation
# -------------------------
rows = []
num_fraud = int(NUM_ROWS * FRAUD_RATIO)
num_legit = NUM_ROWS - num_fraud

labels = ["fraud"] * num_fraud + ["legit"] * num_legit
random.shuffle(labels)

suspicious_device_pool = ["dev-" + short_hash(12) for _ in range(30)]
suspicious_ip_pool = [f"185.62.12.{i}" for i in range(10, 60)]

for i in range(NUM_ROWS):
    is_fraud = labels[i] == "fraud"

    account_external_id = "acct_" + uuid.uuid4().hex
    upload_external_id = "upl_" + uuid.uuid4().hex
    content_ref = str(uuid.uuid4())

    display_name = random_display_name(fake, is_fraud)
    metadata_title = random_title(fake, is_fraud)
    metadata_genre = generate_genre(is_fraud)
    metadata_duration_seconds = generate_duration(is_fraud)
    metadata_bitrate = generate_bitrate(is_fraud)
    metadata_format = generate_format(is_fraud)
    metadata_collaborators = generate_collaborators(fake, is_fraud)
    metadata_album = generate_album(fake, is_fraud)
    metadata_year = generate_year(is_fraud)

    fingerprints_audio_hash = hex_hash(account_external_id + upload_external_id, length=40)
    fingerprints_perceptual_hash = hex_hash(upload_external_id, length=16)

    if is_fraud and RNG.random() < 0.35:
        device_context_device_hash = random.choice(suspicious_device_pool)
    else:
        device_context_device_hash = generate_device_hash(is_fraud)

    if is_fraud and RNG.random() < 0.4:
        device_context_ip = random.choice(suspicious_ip_pool)
    else:
        device_context_ip = generate_ip(is_fraud)

    device_context_user_agent = generate_user_agent(is_fraud)
    account_type = generate_account_type(is_fraud)
    profile_type = generate_profile_type(is_fraud)
    expected_category = "fraud" if is_fraud else "legit"

    row = {
        "account_external_id": account_external_id,
        "account_type": account_type,
        "display_name": display_name,
        "upload_external_id": upload_external_id,
        "metadata_title": metadata_title,
        "metadata_genre": metadata_genre,
        "metadata_duration_seconds": int(metadata_duration_seconds),
        "metadata_bitrate": int(metadata_bitrate),
        "metadata_format": metadata_format,
        "metadata_collaborators": metadata_collaborators,
        "metadata_album": metadata_album,
        "metadata_year": int(metadata_year),
        "content_ref": content_ref,
        "fingerprints_audio_hash": fingerprints_audio_hash,
        "fingerprints_perceptual_hash": fingerprints_perceptual_hash,
        "device_context_device_hash": device_context_device_hash,
        "device_context_ip": device_context_ip,
        "device_context_user_agent": device_context_user_agent,
        "expected_category": expected_category,
        "profile_type": profile_type,
    }

    rows.append(row)

# -------------------------
# Create DataFrame and save CSV
# -------------------------
df = pd.DataFrame(rows, columns=[
    "account_external_id",
    "account_type",
    "display_name",
    "upload_external_id",
    "metadata_title",
    "metadata_genre",
    "metadata_duration_seconds",
    "metadata_bitrate",
    "metadata_format",
    "metadata_collaborators",
    "metadata_album",
    "metadata_year",
    "content_ref",
    "fingerprints_audio_hash",
    "fingerprints_perceptual_hash",
    "device_context_device_hash",
    "device_context_ip",
    "device_context_user_agent",
    "expected_category",
    "profile_type",
])

# Sanity checks
assert len(df) == NUM_ROWS
# pandas Series.is_unique returns a boolean; check directly
assert df["account_external_id"].is_unique, "account_external_id values are not unique"
assert df["upload_external_id"].is_unique, "upload_external_id values are not unique"

# Save to CSV
output_filename = "synthetic_data_gemini.csv"
df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"Synthetic dataset generated: {output_filename} (rows={len(df)})")