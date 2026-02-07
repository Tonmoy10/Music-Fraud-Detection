"""
Ultra-Realistic Synthetic Dataset Generator (v2) for Music Streaming Fraud Detection
-----------------------------------------------------------------------------------

Creates 1000-row CSV: chatgpt_synthetic_music_fraud.csv

Realism upgrades in this version:
  1) Multiple uploads per account (catalog behavior; long-tail distribution).
  2) Album releases with multi-track tracklists + track-number patterns in titles.
  3) Album-level coherence: same year/genre/album name; correlated durations across an album.
  4) Localized IP patterns:
       - Legit accounts: 1–3 "home" /24 blocks per account, devices mostly stay within those blocks,
         occasional roaming IPs.
       - Bot farms: per-farm datacenter /24 blocks reused across many accounts/devices.
       - Hacked accounts: mostly residential blocks but occasional datacenter/proxy-like blocks.
  5) Fraud is stealthy (not cartoonish): some fraud mimics legit quality, titles, and durations.
  6) Fingerprint collisions at plausible rates via shared "assets" and perceptual families,
     especially within bot farms, but with enough entropy to avoid obvious synthetic artifacts.

Dependencies:
  pip install pandas numpy faker

Output columns MUST match the user's schema exactly (kept).
"""

import json
import random
import uuid
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker


# ---------------------------- Configuration ---------------------------- #

SEED = 42
N_ROWS = 1000
OUT_PATH = "synthetic_data_gpt.csv"

GENRES = ["Pop", "Hip-Hop", "Ambient", "Noise"]
FORMATS = ["mp3", "wav", "flac"]
BITRATES = [128, 192, 256, 320]  # 192 added for realism; still fits "Integer" spec.

COMMON_USER_AGENTS = [
    # Desktop browsers
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/122.0.0.0 Safari/537.36",
    # Mobile browsers
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36",
    # App-like clients
    "Spotify/1.2.32 (Mac OS X; Darwin 22.6.0)",
    "Spotify/1.2.32 (Windows 10; Win64; x64)",
    "AppleCoreMedia/1.0.0.21D50 (iPhone; U; CPU OS 17_3 like Mac OS X; en_us)",
    # Occasional automation
    "python-requests/2.31.0",
    "curl/8.1.2",
]

# Token pools to generate plausible titles/albums
TITLE_TOKENS = [
    "midnight", "echo", "neon", "drift", "horizon", "gravity", "atlas", "signal", "afterglow",
    "velvet", "static", "breathe", "orbit", "alchemy", "cascade", "solstice", "ripple", "mirage", "pulse",
    "dream", "city", "shadow", "gold", "river", "electric", "paper", "glass", "moon", "sunset", "blue", "wild"
]
AMBIENT_TOKENS = [
    "rain", "wind", "ocean", "sleep", "calm", "dawn", "night", "fog", "forest", "stars",
    "drone", "pad", "meditation", "stillness", "soft", "waves", "hush", "bloom", "aurora", "ember"
]
NOISE_TOKENS = [
    "texture", "feedback", "distortion", "glitch", "friction", "hiss", "machine", "fragment", "signal", "static"
]
HIPHOP_TOKENS = [
    "street", "rhythm", "flow", "cipher", "bass", "hustle", "vibe", "crew", "mic", "beats", "verse"
]


# ---------------------------- Reproducibility ---------------------------- #

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)


# ---------------------------- Helpers ---------------------------- #

def clamp_int(x, lo, hi) -> int:
    return int(max(lo, min(hi, int(round(x)))))


def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def uuid_hex(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def deterministic_hash(s: str, algo: str = "sha1") -> str:
    h = hashlib.new(algo)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def title_case(words) -> str:
    return " ".join(w.capitalize() for w in words)


def json_list_str(lst) -> str:
    return json.dumps(lst, ensure_ascii=False)


def pick_tokens_for_genre(genre: str, k_min: int, k_max: int):
    k = random.randint(k_min, k_max)
    if genre == "Ambient":
        return random.sample(AMBIENT_TOKENS, k=k)
    if genre == "Noise":
        return random.sample(NOISE_TOKENS, k=k)
    if genre == "Hip-Hop":
        return random.sample(HIPHOP_TOKENS, k=k)
    return random.sample(TITLE_TOKENS, k=k)


def likely_public_ipv4_block(fake: Faker) -> str:
    """
    Take a Faker public IPv4 and return its /24 block (first 3 octets).
    Using Faker keeps blocks plausible and avoids reserved ranges most of the time.
    """
    ip = fake.ipv4_public()
    parts = ip.split(".")
    return ".".join(parts[:3])


def ip_from_block(block: str) -> str:
    # Avoid .0 and .255; use 2..254
    host = random.randint(2, 254)
    return f"{block}.{host}"


# ---------------------------- Account behavior models ---------------------------- #

def sample_account_type(profile_type: str) -> str:
    if profile_type == "normal_user":
        return weighted_choice(["Basic", "Premium", "Artist"], [0.03, 0.07, 0.90])
    if profile_type == "hacked_account":
        return weighted_choice(["Basic", "Premium", "Artist"], [0.10, 0.20, 0.70])
    return weighted_choice(["Basic", "Premium", "Artist"], [0.02, 0.03, 0.95])  # bot_farm


def sample_primary_genre(profile_type: str) -> str:
    if profile_type == "normal_user":
        return weighted_choice(GENRES, [0.38, 0.32, 0.22, 0.08])
    if profile_type == "hacked_account":
        return weighted_choice(GENRES, [0.30, 0.25, 0.30, 0.15])
    return weighted_choice(GENRES, [0.10, 0.10, 0.48, 0.32])  # bot_farm


def sample_genre_for_upload(primary: str, profile_type: str) -> str:
    if profile_type == "normal_user":
        if random.random() < 0.80:
            return primary
        return weighted_choice(GENRES, [0.35, 0.30, 0.25, 0.10])
    if profile_type == "hacked_account":
        if random.random() < 0.72:
            return primary
        return weighted_choice(GENRES, [0.30, 0.25, 0.30, 0.15])
    # bot_farm: can pivot, but still tends to stay in ambient/noise niches
    if random.random() < 0.68:
        return primary
    return weighted_choice(GENRES, [0.10, 0.10, 0.48, 0.32])


def sample_professionalism(account_type: str, profile_type: str) -> float:
    if profile_type == "bot_farm":
        base = np.random.beta(2, 7)
    elif profile_type == "hacked_account":
        base = np.random.beta(3, 4)
    else:
        base = np.random.beta(4, 2)

    if account_type == "Artist":
        base += 0.08
    elif account_type == "Premium":
        base += 0.03
    return float(min(1.0, max(0.0, base)))


def sample_year_profile(profile_type: str) -> int:
    current_year = datetime.utcnow().year
    if profile_type == "bot_farm":
        # Mostly recent, occasional backdating for legitimacy
        options = [current_year - 6, current_year - 5, current_year - 4, current_year - 3,
                   current_year - 2, current_year - 1, current_year]
        probs = np.array([0.02, 0.03, 0.06, 0.10, 0.18, 0.38, 0.23])
        return int(np.random.choice(options, p=probs / probs.sum()))
    if profile_type == "hacked_account":
        options = list(range(2005, current_year + 1))
        w = np.exp((np.array(options) - 2005) / 7.5)
        w = w / w.sum()
        return int(np.random.choice(options, p=w))
    # normal_user: broader, weighted recent
    options = list(range(1995, current_year + 1))
    w = np.exp((np.array(options) - 1995) / 9.0)
    w = w / w.sum()
    return int(np.random.choice(options, p=w))


def sample_format_and_bitrate(year: int, genre: str, professionalism: float, profile_type: str):
    current_year = datetime.utcnow().year
    recency = (year - (current_year - 12)) / 12.0
    recency = max(0.0, min(1.0, recency))

    # Base format probabilities
    p_mp3 = 0.72 - 0.25 * professionalism - 0.10 * recency
    p_flac = 0.10 + 0.22 * professionalism + 0.10 * recency
    p_wav = 1.0 - p_mp3 - p_flac

    # Genre tweak
    if genre in ("Ambient", "Noise"):
        p_wav += 0.05 * professionalism
        p_mp3 -= 0.03 * professionalism

    # Fraud tweaks (but allow mimic)
    if profile_type == "bot_farm":
        p_mp3 += 0.16
        p_flac -= 0.10
        p_wav -= 0.06
    elif profile_type == "hacked_account":
        p_mp3 += 0.06
        p_flac -= 0.04
        p_wav -= 0.02

    probs = np.array([max(0.01, p_mp3), max(0.01, p_wav), max(0.01, p_flac)], dtype=float)
    probs = probs / probs.sum()
    fmt = str(np.random.choice(["mp3", "wav", "flac"], p=probs))

    # Bitrate probabilities
    p128 = 0.28 - 0.18 * professionalism - 0.10 * recency
    p192 = 0.22 - 0.06 * professionalism - 0.04 * recency
    p256 = 0.30 + 0.12 * professionalism + 0.08 * recency
    p320 = 1.0 - (p128 + p192 + p256)

    # Format adjustments
    if fmt in ("wav", "flac"):
        p128 -= 0.10
        p192 -= 0.05
        p256 += 0.07
        p320 += 0.08

    # Fraud adjustments
    if profile_type == "bot_farm":
        p128 += 0.16
        p192 += 0.06
        p256 -= 0.11
        p320 -= 0.11
    elif profile_type == "hacked_account":
        p128 += 0.08
        p192 += 0.03
        p256 -= 0.06
        p320 -= 0.05

    probs = np.array([p128, p192, p256, p320], dtype=float)
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()
    bitrate = int(np.random.choice([128, 192, 256, 320], p=probs))

    return fmt, bitrate


def generate_artist_name(fake: Faker, profile_type: str) -> str:
    # Keep bot farm names mostly plausible
    if profile_type == "bot_farm":
        if random.random() < 0.70:
            w = title_case(fake.words(nb=random.randint(1, 2)))
            suffix = random.choice(["", "", "", " Music", " Studio", " Sounds", " Records"])
            return f"{w}{suffix}".strip()
        return f"{title_case(fake.words(nb=2))} {random.choice(['Audio', 'Lab', 'Network', 'Collective'])}"
    # legit/hacked
    if random.random() < 0.55:
        return fake.name()
    w = title_case(fake.words(nb=random.randint(1, 2)))
    suffix = random.choice(["", "", "", " Official", " Music"])
    return f"{w}{suffix}".strip()


def generate_album_name(fake: Faker, genre: str, profile_type: str) -> str:
    # Bot farms occasionally use generic volume naming (but not always)
    if profile_type == "bot_farm" and random.random() < 0.25:
        return f"{title_case(pick_tokens_for_genre(genre, 1, 2))} Vol. {random.randint(1, 60)}"
    tokens = pick_tokens_for_genre(genre, 2, 4)
    # Small chance of punctuation/formatting like real releases
    if random.random() < 0.08:
        return f"{title_case(tokens)}: {title_case(pick_tokens_for_genre(genre, 1, 2))}"
    return title_case(tokens)


def generate_track_base_title(fake: Faker, genre: str, profile_type: str, idx: int) -> str:
    # Bot farms sometimes templated, but mostly plausible
    if profile_type == "bot_farm" and random.random() < 0.03:
        return f"Track {str(idx).zfill(3)}"

    tokens = pick_tokens_for_genre(genre, 2, 4)
    t = title_case(tokens)

    # Realistic modifiers
    if random.random() < (0.12 if profile_type != "bot_farm" else 0.08):
        t += f" ({random.choice(['Remastered', 'Radio Edit', 'Live', 'Acoustic', 'Instrumental'])})"
    if random.random() < (0.18 if genre in ("Pop", "Hip-Hop") else 0.07):
        t += f" (feat. {fake.first_name()} {fake.last_name()})"

    # Bot farm mild templating
    if profile_type == "bot_farm" and random.random() < 0.16:
        t = f"{t} {random.choice(['Part', 'Session', 'Phase'])} {random.randint(1, 30)}"

    return t


def format_track_title_with_number(base_title: str, track_no: int, total_tracks: int) -> str:
    """
    Real-world-ish track numbering conventions:
      - Some releases prefix "01 " or "1. "
      - Some add "- Track 01"
      - Many don't include track numbers at all
    """
    r = random.random()
    if r < 0.55:
        return base_title
    if r < 0.80:
        return f"{track_no:02d} {base_title}"
    if r < 0.92:
        return f"{track_no}. {base_title}"
    # Occasionally include total tracks (rare)
    return f"{track_no:02d}/{total_tracks:02d} {base_title}"


def generate_collaborators(fake: Faker, genre: str, profile_type: str) -> str:
    if profile_type == "bot_farm":
        if random.random() < 0.90:
            return "[]"
        return json_list_str([random.choice(["Various Artists", "Studio Collaborator", "A. Producer"])])

    base_none = 0.70 if genre in ("Ambient", "Noise") else 0.55
    if random.random() < base_none:
        return "[]"

    k = random.choices([1, 2, 3], weights=[0.70, 0.23, 0.07], k=1)[0]
    names = [fake.name() for _ in range(k)]
    return json_list_str(names)


# ---------------------------- Fraud infrastructure pools ---------------------------- #

class BotFarmPools:
    """
    Shared pools create realistic reuse patterns:
      - Per-farm datacenter /24 blocks (localized IP reuse)
      - Device IDs reused across many bot accounts
      - Content asset IDs and perceptual families shared within a farm
    """
    def __init__(self, fake: Faker, n_farms: int = 7):
        self.fake = fake
        self.n_farms = n_farms

        # Per-farm datacenter /24 blocks (localized). Each farm has a handful of blocks.
        self.farm_blocks = {}
        for farm_id in range(1, n_farms + 1):
            blocks = set()
            while len(blocks) < random.randint(8, 18):
                blocks.add(likely_public_ipv4_block(fake))
            self.farm_blocks[farm_id] = list(blocks)

        # Global "proxy-ish" blocks used occasionally (adds realism)
        self.proxy_blocks = list({likely_public_ipv4_block(fake) for _ in range(60)})

        # Device pool: moderate size to avoid cartoonish repetition
        self.device_pool = [deterministic_hash(f"farm_device_{i}", "md5") for i in range(420)]

        # UA pool: mostly normal UAs, some automation
        self.ua_pool = COMMON_USER_AGENTS[:]

        # Per-farm content pools
        self.farm_audio_assets = {
            farm_id: [f"asset_{farm_id}_{j}" for j in range(520)]
            for farm_id in range(1, n_farms + 1)
        }
        self.farm_phash_families = {
            farm_id: [f"phfam_{farm_id}_{j}" for j in range(360)]
            for farm_id in range(1, n_farms + 1)
        }

    def sample_datacenter_ip(self, farm_id: int) -> str:
        block = random.choice(self.farm_blocks[farm_id])
        return ip_from_block(block)

    def sample_proxy_ip(self) -> str:
        return ip_from_block(random.choice(self.proxy_blocks))


def compute_fingerprints(upload_external_id: str, content_ref: str, asset_id: str = None, phash_family: str = None):
    # Audio hash: collide at asset-level; otherwise unique
    if asset_id is None:
        audio_hash = deterministic_hash(f"{upload_external_id}_{content_ref}", "sha1")
    else:
        audio_hash = deterministic_hash(f"AUDIO::{asset_id}", "sha1")

    # Perceptual hash: collide at family-level; otherwise unique
    if phash_family is None:
        p_hash = deterministic_hash(f"{upload_external_id}_{content_ref}", "md5")
    else:
        p_hash = deterministic_hash(f"PHASH::{phash_family}", "md5")

    return audio_hash, p_hash


# ---------------------------- Album modeling ---------------------------- #

def sample_upload_count(profile_type: str) -> int:
    # Long-tail catalogs; bot farms produce more uploads/account but not absurdly.
    if profile_type == "normal_user":
        r = random.random()
        if r < 0.70:
            return random.randint(1, 3)
        if r < 0.92:
            return random.randint(4, 9)
        return random.randint(10, 22)
    if profile_type == "hacked_account":
        r = random.random()
        if r < 0.55:
            return random.randint(3, 7)
        if r < 0.90:
            return random.randint(8, 16)
        return random.randint(17, 28)
    # bot_farm
    r = random.random()
    if r < 0.40:
        return random.randint(10, 24)
    if r < 0.80:
        return random.randint(25, 50)
    return random.randint(51, 80)


def sample_album_track_count(genre: str, profile_type: str) -> int:
    # Realistic: EPs and LPs; ambient/noise can be longer.
    if profile_type == "bot_farm":
        if genre in ("Ambient", "Noise"):
            return random.randint(12, 30)
        return random.randint(10, 22)
    if genre == "Ambient":
        return random.choice([6, 8, 10, 12, 14, 16, 18])
    if genre == "Noise":
        return random.choice([5, 7, 9, 11, 13, 15])
    # Pop/Hip-Hop
    return random.choice([4, 5, 6, 8, 10, 12, 13, 14])


def album_duration_profile(genre: str, profile_type: str):
    # Returns (mu, sigma, clip_lo, clip_hi) for track durations in the album
    if genre == "Pop":
        mu, sigma, lo, hi = 205, (18 if profile_type != "bot_farm" else 12), 120, 360
    elif genre == "Hip-Hop":
        mu, sigma, lo, hi = 195, (22 if profile_type != "bot_farm" else 14), 110, 420
    elif genre == "Ambient":
        mu, sigma, lo, hi = 330, (65 if profile_type != "bot_farm" else 40), 120, 1200
    else:  # Noise
        mu, sigma, lo, hi = 170, (45 if profile_type != "bot_farm" else 28), 45, 600

    # Bot farms tilt shorter for cost but not always
    if profile_type == "bot_farm" and random.random() < 0.55:
        mu *= 0.75
        lo = max(31, int(lo * 0.8))
    return mu, sigma, lo, hi


def make_album(fake: Faker, account: dict, album_idx: int) -> dict:
    profile_type = account["profile_type"]
    primary_genre = account["primary_genre"]

    genre = primary_genre if random.random() < 0.78 else weighted_choice(GENRES, [0.35, 0.30, 0.25, 0.10])
    year = sample_year_profile(profile_type)
    name = generate_album_name(fake, genre, profile_type)

    # Determine EP/LP-like labeling occasionally (very common in metadata)
    if random.random() < 0.10:
        name = f"{name} EP"
    elif random.random() < 0.06:
        name = f"{name} (Deluxe)"

    track_count = sample_album_track_count(genre, profile_type)
    mu, sigma, lo, hi = album_duration_profile(genre, profile_type)

    # Album-level "sound signature": slight shifts in mean
    mu = mu + np.random.normal(0, 18)

    # Pre-generate track titles/durations
    tracks = []
    for t_no in range(1, track_count + 1):
        base_title = generate_track_base_title(fake, genre, profile_type, idx=(album_idx * 100 + t_no))
        title = format_track_title_with_number(base_title, t_no, track_count)

        dur = clamp_int(np.random.normal(mu, sigma), lo, hi)

        tracks.append({
            "track_no": t_no,
            "title": title,
            "base_title": base_title,
            "duration": dur,
        })

    return {
        "album_name": name,
        "album_year": year,
        "album_genre": genre,
        "track_count": track_count,
        "tracks": tracks,
    }


def decide_release_plan(profile_type: str, n_uploads: int):
    """
    Decide how many albums exist and how many uploads are singles.
    Returns (albums_count, p_album_track).
    """
    if profile_type == "bot_farm":
        # Bot farms often do "albums" with many tracks, but also dump singles.
        albums_count = random.choices([0, 1, 2, 3, 4], weights=[0.10, 0.25, 0.28, 0.22, 0.15], k=1)[0]
        p_album_track = 0.55
    elif profile_type == "hacked_account":
        albums_count = random.choices([0, 1, 2, 3], weights=[0.25, 0.38, 0.25, 0.12], k=1)[0]
        p_album_track = 0.50
    else:
        albums_count = random.choices([0, 1, 2, 3], weights=[0.35, 0.38, 0.20, 0.07], k=1)[0]
        p_album_track = 0.60

    # Limit albums to what uploads can support
    albums_count = min(albums_count, max(0, n_uploads // 4))
    return albums_count, p_album_track


# ---------------------------- Device/IP behavior ---------------------------- #

def sample_user_agent(profile_type: str) -> str:
    """
    COMMON_USER_AGENTS currently has 11 entries.
    These weights MUST have length 11 to avoid ValueError.
    The last two entries are automation-like (python-requests, curl).
    """
    if profile_type == "bot_farm":
        # Mostly normal-looking clients, some automation leakage (still low)
        weights = [0.16, 0.10, 0.10, 0.08, 0.10, 0.08, 0.10, 0.08, 0.05, 0.025, 0.025]
        return weighted_choice(COMMON_USER_AGENTS, weights)

    if profile_type == "hacked_account":
        # Similar to normal, slightly higher chance of odd clients
        weights = [0.18, 0.12, 0.10, 0.08, 0.12, 0.10, 0.12, 0.10, 0.04, 0.02, 0.02]
        return weighted_choice(COMMON_USER_AGENTS, weights)

    # normal_user
    weights = [0.18, 0.14, 0.10, 0.09, 0.12, 0.10, 0.11, 0.10, 0.03, 0.015, 0.015]
    return weighted_choice(COMMON_USER_AGENTS, weights)


def make_residential_blocks(fake: Faker, profile_type: str) -> list:
    # Per-account localized /24 blocks
    n_blocks = random.choices([1, 2, 3, 4], weights=[0.52, 0.30, 0.14, 0.04], k=1)[0]
    # Hacked accounts may have more churn
    if profile_type == "hacked_account" and random.random() < 0.35:
        n_blocks = min(5, n_blocks + 1)

    blocks = set()
    while len(blocks) < n_blocks:
        blocks.add(likely_public_ipv4_block(fake))
    return list(blocks)


def assign_devices(profile_type: str) -> int:
    if profile_type == "bot_farm":
        return random.randint(1, 2)
    return random.choices([1, 2, 3, 4], weights=[0.55, 0.28, 0.13, 0.04], k=1)[0]


def build_device_entries(fake: Faker, account: dict, farm_pools: BotFarmPools):
    profile_type = account["profile_type"]
    n_devices = account["n_devices"]

    device_entries = []
    if profile_type == "bot_farm":
        # Bot farms: devices from shared pool; per-device IP blocks are datacenter blocks
        for _ in range(n_devices):
            device_hash = random.choice(farm_pools.device_pool)
            ua = sample_user_agent(profile_type)
            # Each device sticks to a subset of farm blocks
            blocks = random.sample(farm_pools.farm_blocks[account["farm_id"]], k=random.randint(2, 6))
            device_entries.append({"device_hash": device_hash, "ua": ua, "blocks": blocks})
        return device_entries

    # Legit/hacked: unique-ish devices; residential blocks
    for _ in range(n_devices):
        device_hash = deterministic_hash(f"{account['account_external_id']}::{uuid.uuid4().hex}", "md5")
        ua = sample_user_agent(profile_type)
        # Each device uses a subset of the account's residential blocks
        blocks = random.sample(account["res_blocks"], k=random.randint(1, min(3, len(account["res_blocks"]))))
        device_entries.append({"device_hash": device_hash, "ua": ua, "blocks": blocks})
    return device_entries


def sample_ip_for_upload(fake: Faker, account: dict, device_entry: dict, farm_pools: BotFarmPools) -> str:
    profile_type = account["profile_type"]

    # Base: choose from device's localized blocks
    def from_device_blocks():
        return ip_from_block(random.choice(device_entry["blocks"]))

    if profile_type == "normal_user":
        # Mostly stable within localized blocks; occasional roam
        if random.random() < 0.80:
            return from_device_blocks()
        # roam: new public IP (travel, mobile, VPN)
        return fake.ipv4_public()

    if profile_type == "hacked_account":
        # Mix of residential and suspicious IPs
        r = random.random()
        if r < 0.65:
            return from_device_blocks()
        if r < 0.80:
            # roam
            return fake.ipv4_public()
        # suspicious: datacenter/proxy-like sometimes
        if random.random() < 0.65:
            return farm_pools.sample_proxy_ip()
        return farm_pools.sample_datacenter_ip(farm_id=random.choice(list(farm_pools.farm_blocks.keys())))

    # bot_farm
    r = random.random()
    if r < 0.88:
        return from_device_blocks()
    # Occasionally switch blocks to mimic distributed infra
    if random.random() < 0.55:
        return farm_pools.sample_datacenter_ip(account["farm_id"])
    return farm_pools.sample_proxy_ip()


# ---------------------------- Account construction ---------------------------- #

def build_accounts(fake: Faker, target_rows: int, farm_pools: BotFarmPools):
    """
    Build enough accounts so sum(upload_count) >= target_rows (with buffer).
    Note: fraud share is driven by higher upload volume per bot account (realistic).
    """
    accounts = []
    total_uploads = 0

    while total_uploads < target_rows * 1.25:
        profile_type = weighted_choice(
            ["normal_user", "bot_farm", "hacked_account"],
            [0.84, 0.11, 0.05]
        )

        account_external_id = uuid_hex("acc_")
        account_type = sample_account_type(profile_type)
        display_name = generate_artist_name(fake, profile_type)
        primary_genre = sample_primary_genre(profile_type)
        professionalism = sample_professionalism(account_type, profile_type)
        upload_count = sample_upload_count(profile_type)
        n_devices = assign_devices(profile_type)

        farm_id = None
        res_blocks = None
        if profile_type == "bot_farm":
            farm_id = random.randint(1, farm_pools.n_farms)
        else:
            res_blocks = make_residential_blocks(fake, profile_type)

        accounts.append({
            "account_external_id": account_external_id,
            "account_type": account_type,
            "display_name": display_name,
            "profile_type": profile_type,
            "expected_category": "fraud" if profile_type in ("bot_farm", "hacked_account") else "legit",
            "primary_genre": primary_genre,
            "professionalism": professionalism,
            "upload_count": upload_count,
            "n_devices": n_devices,
            "farm_id": farm_id,
            "res_blocks": res_blocks,
        })

        total_uploads += upload_count

    random.shuffle(accounts)
    return accounts


# ---------------------------- Row generation ---------------------------- #

def generate_rows(fake: Faker, accounts: list, farm_pools: BotFarmPools, target_rows: int):
    rows = []
    current_year = datetime.utcnow().year

    for account in accounts:
        if len(rows) >= target_rows:
            break

        profile_type = account["profile_type"]
        expected_category = account["expected_category"]

        # Albums & release plan
        n_uploads = account["upload_count"]
        albums_count, p_album_track = decide_release_plan(profile_type, n_uploads)

        albums = [make_album(fake, account, album_idx=i + 1) for i in range(albums_count)]

        # Device entries for this account
        device_entries = build_device_entries(fake, account, farm_pools)

        # Fingerprint reuse probabilities (subtle, not extreme)
        if profile_type == "normal_user":
            p_reuse_asset = 0.03
            p_reuse_phash = 0.06
        elif profile_type == "hacked_account":
            p_reuse_asset = 0.08
            p_reuse_phash = 0.12
        else:
            p_reuse_asset = 0.20
            p_reuse_phash = 0.28

        # Local pools per account (even bot farms have some per-account uniqueness)
        local_asset_pool = []
        local_phash_pool = []

        # Maintain album track pointers so we emit multi-track albums
        album_track_queues = []
        for alb in albums:
            # Make a queue of tracks; we’ll pull from it
            album_track_queues.append({
                "album": alb,
                "remaining": alb["tracks"][:],  # list of dicts
            })

        for _ in range(n_uploads):
            if len(rows) >= target_rows:
                break

            upload_external_id = uuid_hex("upl_")
            content_ref = str(uuid.uuid4())

            # Choose whether this upload is an album track vs single
            use_album = bool(album_track_queues) and (random.random() < p_album_track)

            if use_album:
                # Choose an album queue that still has tracks
                choices = [q for q in album_track_queues if q["remaining"]]
                if not choices:
                    use_album = False
                else:
                    q = random.choice(choices)
                    alb = q["album"]
                    track = q["remaining"].pop(0)  # sequential tracklist output

                    metadata_album = alb["album_name"]
                    metadata_year = alb["album_year"]
                    metadata_genre = alb["album_genre"]

                    metadata_title = track["title"]
                    metadata_duration_seconds = int(track["duration"])

                    # Album tracks often share same format/bitrate tendencies
                    metadata_format, metadata_bitrate = sample_format_and_bitrate(
                        year=metadata_year,
                        genre=metadata_genre,
                        professionalism=account["professionalism"],
                        profile_type=profile_type
                    )

                    # Occasional album re-release/remaster: same album name, different year,
                    # slightly different title suffix and/or fingerprints
                    if profile_type != "bot_farm" and random.random() < 0.03:
                        metadata_year = min(current_year, metadata_year + random.choice([1, 2, 3]))
                        if "(Remastered" not in metadata_title and random.random() < 0.5:
                            metadata_title = f"{metadata_title} (Remastered)"

            if not use_album:
                # Single
                metadata_genre = sample_genre_for_upload(account["primary_genre"], profile_type)
                metadata_year = sample_year_profile(profile_type)
                base_title = generate_track_base_title(fake, metadata_genre, profile_type, idx=len(rows) + 1)
                metadata_title = base_title

                # Realistic single "album" field behavior:
                # sometimes "Single", sometimes a named "single-era" collection
                metadata_album = weighted_choice(
                    ["Single", generate_album_name(fake, metadata_genre, profile_type)],
                    [0.62, 0.38]
                )

                # Duration: genre/profile-driven
                # More realism: occasionally interludes, long ambient loops, etc.
                if profile_type == "bot_farm":
                    # stealthy mixture
                    r = random.random()
                    if r < 0.40:
                        metadata_duration_seconds = clamp_int(np.random.normal(40, 8), 31, 85)
                    elif r < 0.82:
                        mu = 210 if metadata_genre in ("Pop", "Hip-Hop") else 280
                        metadata_duration_seconds = clamp_int(np.random.normal(mu, 60), 90, 600)
                    else:
                        metadata_duration_seconds = clamp_int(np.random.normal(520, 150), 240, 1200)
                else:
                    # normal/hacked
                    if metadata_genre == "Pop":
                        metadata_duration_seconds = clamp_int(np.random.normal(205, 35), 120, 360)
                    elif metadata_genre == "Hip-Hop":
                        metadata_duration_seconds = clamp_int(np.random.normal(195, 40), 110, 420)
                    elif metadata_genre == "Ambient":
                        metadata_duration_seconds = clamp_int(np.random.normal(330, 110), 120, 1200)
                    else:
                        metadata_duration_seconds = clamp_int(np.random.normal(170, 70), 45, 600)

                metadata_format, metadata_bitrate = sample_format_and_bitrate(
                    year=metadata_year,
                    genre=metadata_genre,
                    professionalism=account["professionalism"],
                    profile_type=profile_type
                )

            # Inject rare outliers to avoid overly clean separability
            if expected_category == "legit" and random.random() < 0.02:
                # short skit/intro
                metadata_duration_seconds = clamp_int(np.random.normal(75, 18), 30, 140)
                metadata_bitrate = weighted_choice([128, 192, 256], [0.50, 0.30, 0.20])
                metadata_format = "mp3"

            if expected_category == "fraud" and random.random() < 0.06:
                # mimic "high quality"
                metadata_duration_seconds = clamp_int(np.random.normal(230, 45), 120, 480)
                metadata_bitrate = weighted_choice([256, 320], [0.55, 0.45])
                metadata_format = weighted_choice(["mp3", "flac", "wav"], [0.70, 0.20, 0.10])

            # Collaborators
            metadata_collaborators = generate_collaborators(fake, metadata_genre, profile_type)

            # Fingerprint collisions: asset_id & perceptual family decisions
            asset_id = None
            phash_family = None

            if profile_type == "bot_farm":
                farm_id = account["farm_id"]

                # Asset reuse: mix of farm-shared and local
                if random.random() < p_reuse_asset and (local_asset_pool or farm_id is not None):
                    if local_asset_pool and random.random() < 0.40:
                        asset_id = random.choice(local_asset_pool)
                    else:
                        asset_id = random.choice(farm_pools.farm_audio_assets[farm_id])
                else:
                    if farm_id is not None and random.random() < 0.72:
                        asset_id = random.choice(farm_pools.farm_audio_assets[farm_id])
                    else:
                        asset_id = f"unique_asset_{account['account_external_id']}_{uuid.uuid4().hex}"
                    local_asset_pool.append(asset_id)

                # Perceptual family reuse
                if random.random() < p_reuse_phash and (local_phash_pool or farm_id is not None):
                    if local_phash_pool and random.random() < 0.50:
                        phash_family = random.choice(local_phash_pool)
                    else:
                        phash_family = random.choice(farm_pools.farm_phash_families[farm_id])
                else:
                    if farm_id is not None and random.random() < 0.78:
                        phash_family = random.choice(farm_pools.farm_phash_families[farm_id])
                    else:
                        phash_family = f"unique_phfam_{account['account_external_id']}_{uuid.uuid4().hex}"
                    local_phash_pool.append(phash_family)

            else:
                # legit/hacked: mostly unique; small reupload/remaster probability
                if random.random() < p_reuse_asset and local_asset_pool:
                    asset_id = random.choice(local_asset_pool)
                else:
                    asset_id = f"asset_{account['account_external_id']}_{uuid.uuid4().hex}"
                    local_asset_pool.append(asset_id)

                if random.random() < p_reuse_phash and local_phash_pool:
                    phash_family = random.choice(local_phash_pool)
                else:
                    phash_family = f"phfam_{account['account_external_id']}_{uuid.uuid4().hex}"
                    local_phash_pool.append(phash_family)

            fingerprints_audio_hash, fingerprints_perceptual_hash = compute_fingerprints(
                upload_external_id=upload_external_id,
                content_ref=content_ref,
                asset_id=asset_id,
                phash_family=phash_family
            )

            # Device context: choose a device entry
            dev = random.choice(device_entries)
            device_context_device_hash = dev["device_hash"]
            device_context_user_agent = dev["ua"]

            # Hacked accounts: occasional UA anomalies (session hijack / remote upload)
            if profile_type == "hacked_account" and random.random() < 0.08:
                device_context_user_agent = sample_user_agent(profile_type)

            device_context_ip = sample_ip_for_upload(fake, account, dev, farm_pools)

            rows.append({
                "account_external_id": account["account_external_id"],
                "account_type": account["account_type"],
                "display_name": account["display_name"],
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
            })

    # If under-shot, build more accounts and continue (rare)
    return rows


def main():
    seed_everything(SEED)
    fake = Faker()

    farm_pools = BotFarmPools(fake=fake, n_farms=7)

    # Build accounts with buffer
    accounts = build_accounts(fake, target_rows=N_ROWS, farm_pools=farm_pools)

    rows = generate_rows(fake, accounts, farm_pools, target_rows=N_ROWS)

    # Top-up if needed
    while len(rows) < N_ROWS:
        extra_accounts = build_accounts(fake, target_rows=(N_ROWS - len(rows)) + 250, farm_pools=farm_pools)
        rows.extend(generate_rows(fake, extra_accounts, farm_pools, target_rows=(N_ROWS - len(rows))))
        rows = rows[:N_ROWS]

    # Shuffle to remove ordering artifacts
    random.shuffle(rows)

    # Exact schema & order required
    columns = [
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
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Integrity checks
    assert len(df) == N_ROWS, "Row count mismatch"
    assert df["upload_external_id"].is_unique, "upload_external_id must be unique"
    assert df["content_ref"].is_unique, "content_ref must be unique"
    assert df["account_external_id"].nunique() < len(df), "Expected multiple uploads per account for realism"
    assert set(df.columns) == set(columns), "Schema mismatch"

    # Save
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    # Minimal console summary
    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print("\nLabel distribution:")
    print(df["expected_category"].value_counts(dropna=False))
    print("\nProfile distribution:")
    print(df["profile_type"].value_counts(dropna=False))
    print("\nAccounts:", df["account_external_id"].nunique())
    print("\nExample rows:")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
