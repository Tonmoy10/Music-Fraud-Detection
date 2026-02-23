"""
Ultra-Realistic Synthetic Dataset Generator (V5-Tuned) for Music Streaming Fraud Detection
----------------------------------------------------------------------------------------
Tuning goals vs V5:
  - Reduce "bitrate always 320" artifact (especially for lossless)
  - Increase realistic legacy MP3 ingestion for older years / less strict pipelines
  - Keep smart bots stealthy (diversify tiers instead of always maxing)
  - Maintain scale stability (chunked writing, scale-aware pools, no boundary pile-ups)

Output: chatgpt_synthetic_music_fraud.csv
"""

import os
import json
import uuid
import math
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------- Configuration ---------------------------- #

SEED = 42
N_ROWS = 5000          # Expandable: set to 100000
CHUNK_SIZE = 5000

OUTPUT_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"
OUT_FILENAME = "dataset_gpt_5000_V5.1.csv"
OUT_PATH = os.path.join(OUTPUT_DIR, OUT_FILENAME)

EARLIEST_YEAR = 1999

COLUMNS = [
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

GENRES = ["Pop", "Hip-Hop", "Ambient", "Noise"]
FORMATS = ["mp3", "wav", "flac"]
BITRATES = [128, 192, 256, 320]
ACCOUNT_TYPES = ["Basic", "Premium", "Artist"]
PROFILE_TYPES = ["normal_user", "bot_farm", "hacked_account"]

# ---------------------------- Reproducibility ---------------------------- #

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

# ---------------------------- Helpers ---------------------------- #

def clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(x)))))


def weighted_choice(options: List[str], weights: List[float]) -> str:
    if len(options) != len(weights):
        raise ValueError(f"weights length {len(weights)} != options length {len(options)}")
    return random.choices(options, weights=weights, k=1)[0]


def uuid_hex(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def h_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def h_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def json_list_str(lst: List[str]) -> str:
    return json.dumps(lst, ensure_ascii=False)


def now_year() -> int:
    return datetime.utcnow().year


def ipv4_block_24(fake: Faker) -> str:
    ip = fake.ipv4_public()
    a, b, c, _ = ip.split(".")
    return f"{a}.{b}.{c}"


def ip_from_block(block_24: str) -> str:
    return f"{block_24}.{random.randint(2, 254)}"


def title_case(words: List[str]) -> str:
    return " ".join(w.capitalize() for w in words if w)


def truncated_int_sampler(draw_fn, lo: int, hi: int, max_tries: int = 40) -> int:
    for _ in range(max_tries):
        v = int(round(draw_fn()))
        if lo <= v <= hi:
            return v
    return int(np.random.randint(lo, hi + 1))

# ---------------------------- Token pools ---------------------------- #

TOKENS_COMMON = [
    "midnight", "echo", "neon", "drift", "horizon", "gravity", "atlas", "signal", "afterglow",
    "velvet", "static", "breathe", "orbit", "alchemy", "cascade", "solstice", "ripple", "mirage", "pulse",
    "dream", "city", "shadow", "gold", "river", "electric", "paper", "glass", "moon", "sunset", "blue", "wild",
    "memory", "thread", "summer", "winter", "stone", "ember", "nova", "paradox", "lullaby", "arcade",
]
TOKENS_AMBIENT = [
    "rain", "wind", "ocean", "sleep", "calm", "dawn", "night", "fog", "forest", "stars",
    "drone", "pad", "meditation", "stillness", "waves", "hush", "aurora", "bloom", "glow", "tide",
]
TOKENS_NOISE = [
    "texture", "feedback", "distortion", "glitch", "friction", "hiss", "machine", "fragment", "signal", "static",
    "overload", "grain", "noise", "shard", "circuit", "rupture",
]
TOKENS_HIPHOP = [
    "street", "rhythm", "flow", "cipher", "bass", "hustle", "vibe", "crew", "mic", "beats", "verse",
    "blocks", "tempo", "hook", "bars", "nightshift",
]


def tokens_for_genre(genre: str) -> List[str]:
    if genre == "Ambient":
        return TOKENS_AMBIENT
    if genre == "Noise":
        return TOKENS_NOISE
    if genre == "Hip-Hop":
        return TOKENS_HIPHOP
    return TOKENS_COMMON

# ---------------------------- Pipelines / policies ---------------------------- #

PIPELINES = [
    "creator_web_portal",
    "desktop_uploader_app",
    "distributor_batch_api",
    "label_ingestion_portal",
    "partner_api",
]

@dataclass
class PipelinePolicy:
    name: str
    lossless_preference: float
    lossless_required_prob: float
    metadata_strictness: float
    ua_profile: str
    session_burstiness: float


PIPELINE_POLICIES: Dict[str, PipelinePolicy] = {
    "creator_web_portal": PipelinePolicy("creator_web_portal", 0.46, 0.09, 0.55, "web", 0.56),
    "desktop_uploader_app": PipelinePolicy("desktop_uploader_app", 0.56, 0.13, 0.66, "desktop", 0.64),
    "distributor_batch_api": PipelinePolicy("distributor_batch_api", 0.76, 0.33, 0.80, "batch", 0.84),
    "label_ingestion_portal": PipelinePolicy("label_ingestion_portal", 0.86, 0.50, 0.88, "label", 0.80),
    "partner_api": PipelinePolicy("partner_api", 0.80, 0.40, 0.82, "partner", 0.76),
}

UA_POOLS: Dict[str, List[str]] = {
    "web": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/122.0.0.0 Safari/537.36",
        "CreatorDashboard/2.1 (WebKit; Upload)",
        "MusicCreatorPortal/3.4 (Web; UploadFlow)",
        "python-requests/2.31.0",
        "curl/8.1.2",
    ],
    "desktop": [
        "MusicUploaderDesktop/5.2 (Windows 10; Win64; x64)",
        "MusicUploaderDesktop/5.2 (Mac OS X 13.6; x64)",
        "CreatorSuite/4.8 (Windows; Upload)",
        "CreatorSuite/4.8 (Mac; Upload)",
        "python-requests/2.31.0",
        "curl/8.1.2",
    ],
    "batch": [
        "DistributorBatchUploader/3.0 (Batch; APIIngest)",
        "IngestionWorker/1.7 (Batch; RetryQueue)",
        "PartnerIngest/2.2 (Batch; MetadataSync)",
        "python-requests/2.31.0",
        "curl/8.1.2",
    ],
    "label": [
        "LabelIngestionPortal/6.1 (Web; ComplianceCheck)",
        "LabelIngestionPortal/6.1 (Web; ReleaseManager)",
        "LabelOpsTool/3.3 (Desktop; Upload)",
        "DistributorBatchUploader/3.0 (Batch; APIIngest)",
        "python-requests/2.31.0",
        "curl/8.1.2",
    ],
    "partner": [
        "PartnerAPIClient/4.1 (Ingest; Audio)",
        "PartnerAPIClient/4.0 (Ingest; Metadata)",
        "PartnerIngest/2.2 (Batch; MetadataSync)",
        "IngestionWorker/1.7 (Batch; RetryQueue)",
        "python-requests/2.31.0",
        "curl/8.1.2",
    ],
}


def sample_user_agent(profile_type: str, policy: PipelinePolicy, smartness: float = 0.0, compromised: bool = False) -> str:
    pool = UA_POOLS[policy.ua_profile]
    n = len(pool)
    base_auto = 0.010
    if profile_type == "bot_farm":
        base_auto = 0.020 * (1.0 - 0.75 * smartness)
        base_auto = max(0.003, base_auto)
    if compromised:
        base_auto = max(base_auto, 0.050)

    human_count = max(1, n - 2)
    w_human = (1.0 - 2 * base_auto) / human_count
    weights = [w_human] * human_count + [base_auto, base_auto]

    if policy.ua_profile in ("web", "desktop") and human_count >= 3:
        bump = 0.06
        weights[0] += bump
        weights[1] += bump * 0.6
        take = bump * 1.6
        for i in range(2, human_count):
            weights[i] = max(1e-6, weights[i] - take / max(1, (human_count - 2)))

    s = sum(weights)
    weights = [w / s for w in weights]
    return weighted_choice(pool, weights)

# ---------------------------- Bot strategies / farms ---------------------------- #

@dataclass
class BotStrategy:
    name: str
    infra_style: str
    volume_profile: str
    reuse_mode: str
    compliance_mimic: float
    low_and_slow: float


BOT_STRATEGIES: List[BotStrategy] = [
    BotStrategy("low_cost_loop",     "datacenter",        "high",   "high_exact", 0.20, 0.10),
    BotStrategy("sharded_variants",  "datacenter",        "high",   "high_phash", 0.32, 0.20),
    BotStrategy("label_mimic",       "datacenter",        "medium", "low_reuse",  0.72, 0.35),
    BotStrategy("compliance_mimic",  "residential_proxy", "medium", "low_reuse",  0.88, 0.55),
    BotStrategy("low_and_slow",      "residential_proxy", "low",    "low_reuse",  0.80, 0.85),
]


def choose_n_farms(n_rows: int) -> int:
    return max(6, int(round(min(55, max(6, math.sqrt(n_rows) / 6.5)))))


@dataclass
class Farm:
    farm_id: int
    strategy: BotStrategy
    datacenter_blocks_24: List[str]
    proxy_blocks_24: List[str]
    residential_proxy_blocks_24: List[str]
    device_pool: List[str]
    asset_pool: List[str]
    phash_families: List[str]


def build_farms(fake: Faker, n_rows: int) -> Dict[int, Farm]:
    n_farms = choose_n_farms(n_rows)
    global_proxy_blocks = list({ipv4_block_24(fake) for _ in range(clamp_int(n_rows * 0.002, 250, 15000))})

    farms: Dict[int, Farm] = {}
    for fid in range(1, n_farms + 1):
        strat = random.choice(BOT_STRATEGIES)

        blocks_per_farm = clamp_int(12 + (math.sqrt(n_rows) / 5.8) + np.random.normal(0, 2), 12, 65)
        datacenter = [ipv4_block_24(fake) for _ in range(blocks_per_farm)]
        proxy_k = clamp_int(6 + (math.sqrt(n_rows) / 9.0) + np.random.normal(0, 1), 5, 28)
        proxy = random.sample(global_proxy_blocks, k=min(len(global_proxy_blocks), proxy_k))

        if strat.infra_style == "residential_proxy":
            resi_pool_size = clamp_int((n_rows / max(1, n_farms)) * 0.35, 800, 40000)
            residential = [ipv4_block_24(fake) for _ in range(resi_pool_size)]
        else:
            residential = []

        base_dev = (n_rows / max(1, n_farms)) * 0.06
        if strat.infra_style == "residential_proxy":
            base_dev *= 2.0
        dev_pool_size = clamp_int(base_dev * np.random.uniform(0.8, 1.7), 250, 12000)
        device_pool = [h_md5(f"farm{fid}_dev_{i}_{uuid.uuid4().hex}")[:16] for i in range(dev_pool_size)]

        asset_pool_size = clamp_int((n_rows / max(1, n_farms)) * np.random.uniform(0.35, 0.85), 900, 25000)
        phash_pool_size = clamp_int((n_rows / max(1, n_farms)) * np.random.uniform(0.28, 0.75), 800, 20000)
        asset_pool = [h_sha1(f"asset::{fid}::{i}::{uuid.uuid4().hex}") for i in range(asset_pool_size)]
        phash_families = [h_sha1(f"phfam::{fid}::{i}::{uuid.uuid4().hex}") for i in range(phash_pool_size)]

        farms[fid] = Farm(fid, strat, datacenter, proxy, residential, device_pool, asset_pool, phash_families)

    return farms

# ---------------------------- Accounts ---------------------------- #

@dataclass
class Device:
    device_hash: str
    home_blocks_24: List[str]


@dataclass
class Account:
    account_external_id: str
    profile_type: str
    expected_category: str
    account_type: str
    display_name: str

    pipeline: str
    policy: PipelinePolicy
    lossless_required: bool
    sophistication: float
    smartness: float
    genre_probs: np.ndarray

    home_blocks_24: List[str]
    devices: List[Device]

    compromised_fraction: float
    compromised_sessions: int

    farm_id: Optional[int]
    bot_strategy: Optional[BotStrategy]
    bot_home_blocks_24: List[str]


def sample_profile_type() -> str:
    return weighted_choice(PROFILE_TYPES, [0.84, 0.11, 0.05])


def expected_category_from_profile(profile_type: str) -> str:
    return "legit" if profile_type == "normal_user" else "fraud"


def sample_sophistication(profile_type: str) -> float:
    if profile_type == "bot_farm":
        return float(np.clip(np.random.beta(2.2, 5.8) + np.random.normal(0, 0.04), 0, 1))
    if profile_type == "hacked_account":
        return float(np.clip(np.random.beta(3.2, 3.6) + np.random.normal(0, 0.04), 0, 1))
    return float(np.clip(np.random.beta(4.2, 2.0) + np.random.normal(0, 0.04), 0, 1))


def sample_account_type(soph: float, profile_type: str) -> str:
    if profile_type == "bot_farm":
        w_artist = 0.56 + 0.30 * soph
        w_premium = 0.10 + 0.10 * soph
    elif profile_type == "hacked_account":
        w_artist = 0.48 + 0.32 * soph
        w_premium = 0.18 + 0.16 * soph
    else:
        w_artist = 0.52 + 0.36 * soph
        w_premium = 0.20 + 0.14 * soph
    w_basic = max(0.02, 1.0 - (w_artist + w_premium))
    return weighted_choice(ACCOUNT_TYPES, [w_basic, w_premium, w_artist])


def sample_pipeline(profile_type: str, account_type: str, soph: float, strategy: Optional[BotStrategy]) -> str:
    # Tuned: slightly more web/desktop to increase plausible MP3 / legacy deliveries.
    if profile_type == "normal_user":
        base = {"creator_web_portal": 0.26, "desktop_uploader_app": 0.20, "distributor_batch_api": 0.32, "label_ingestion_portal": 0.14, "partner_api": 0.08}
    elif profile_type == "hacked_account":
        base = {"creator_web_portal": 0.27, "desktop_uploader_app": 0.18, "distributor_batch_api": 0.30, "label_ingestion_portal": 0.15, "partner_api": 0.10}
    else:
        base = {"creator_web_portal": 0.16, "desktop_uploader_app": 0.12, "distributor_batch_api": 0.38, "label_ingestion_portal": 0.20, "partner_api": 0.14}
        if strategy and strategy.compliance_mimic > 0.8:
            base["label_ingestion_portal"] += 0.10
            base["distributor_batch_api"] += 0.05
            base["creator_web_portal"] = max(0.01, base["creator_web_portal"] - 0.09)

    strict_boost = 0.10 * (0.6 * soph + (0.4 if account_type == "Artist" else 0.0))
    base["label_ingestion_portal"] += strict_boost
    base["distributor_batch_api"] += strict_boost * 0.7
    base["creator_web_portal"] = max(0.01, base["creator_web_portal"] - strict_boost * 0.8)
    base["desktop_uploader_app"] = max(0.01, base["desktop_uploader_app"] - strict_boost * 0.4)

    keys = list(base.keys())
    w = np.array([base[k] for k in keys], dtype=float)
    w = w / w.sum()
    return str(np.random.choice(keys, p=w))


def sample_lossless_required(policy: PipelinePolicy, soph: float, account_type: str, profile_type: str, strategy: Optional[BotStrategy]) -> bool:
    p = policy.lossless_required_prob + 0.14 * soph + (0.06 if account_type == "Artist" else 0.0)
    if profile_type == "bot_farm" and strategy:
        p += 0.25 * strategy.compliance_mimic
    return random.random() < min(0.94, max(0.02, p))


def sample_genre_probs(profile_type: str, strategy: Optional[BotStrategy]) -> np.ndarray:
    if profile_type == "bot_farm":
        if strategy and strategy.compliance_mimic > 0.8:
            alpha = np.array([1.7, 1.6, 1.5, 1.2])
        else:
            alpha = np.array([1.1, 1.0, 2.0, 1.9])
    elif profile_type == "hacked_account":
        alpha = np.array([1.7, 1.6, 1.5, 1.2])
    else:
        alpha = np.array([2.1, 1.9, 1.3, 0.8])
    return np.random.dirichlet(alpha)


def make_display_name(fake: Faker, profile_type: str) -> str:
    if profile_type == "bot_farm" and random.random() < 0.30:
        return f"{random.choice(TOKENS_COMMON).capitalize()}{random.randint(10, 999)}"
    if random.random() < 0.45:
        return fake.user_name().replace("_", " ").title()
    return f"{fake.first_name()} {fake.last_name()}"


def sample_home_blocks(fake: Faker, profile_type: str) -> List[str]:
    if profile_type == "bot_farm":
        return []
    if profile_type == "hacked_account":
        k = int(np.random.choice([1, 2, 3, 4], p=[0.30, 0.36, 0.22, 0.12]))
    else:
        k = int(np.random.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
    return [ipv4_block_24(fake) for _ in range(k)]


def sample_num_devices(profile_type: str, policy: PipelinePolicy, strategy: Optional[BotStrategy]) -> int:
    if profile_type == "bot_farm":
        if strategy and strategy.infra_style == "residential_proxy":
            return int(np.random.choice([1, 2, 2, 3, 3, 4], p=[0.10, 0.24, 0.22, 0.18, 0.16, 0.10]))
        return int(np.random.choice([1, 1, 2, 2, 3], p=[0.32, 0.26, 0.20, 0.16, 0.06]))
    if profile_type == "hacked_account":
        return int(np.random.choice([1, 2, 2, 3, 4], p=[0.25, 0.33, 0.18, 0.16, 0.08]))
    base = int(np.random.choice([1, 2, 2, 3, 4], p=[0.20, 0.36, 0.20, 0.16, 0.08]))
    if policy.ua_profile in ("batch", "label", "partner") and random.random() < 0.65:
        base = max(1, base - 1)
    return base


def build_devices(account_id: str, home_blocks: List[str], n_devices: int) -> List[Device]:
    out: List[Device] = []
    for _ in range(n_devices):
        dev_hash = h_md5(f"{account_id}::{uuid.uuid4().hex}")[:16]
        blocks = random.sample(home_blocks, k=random.randint(1, min(2, len(home_blocks)))) if home_blocks else []
        out.append(Device(dev_hash, blocks))
    return out


def sample_upload_count(profile_type: str, strategy: Optional[BotStrategy]) -> int:
    if profile_type == "normal_user":
        x = np.random.lognormal(mean=0.85, sigma=0.85)
        return clamp_int(x, 1, 65)
    if profile_type == "hacked_account":
        x = np.random.lognormal(mean=1.25, sigma=0.75)
        return clamp_int(x, 2, 95)
    if strategy and strategy.volume_profile == "low":
        x = np.random.lognormal(mean=1.00, sigma=0.60)
        return clamp_int(x, 2, 45)
    if strategy and strategy.volume_profile == "medium":
        x = np.random.lognormal(mean=1.55, sigma=0.60)
        return clamp_int(x, 6, 110)
    x = np.random.lognormal(mean=2.05, sigma=0.55)
    return clamp_int(x, 10, 220)

# ---------------------------- Years (expanded) ---------------------------- #

def choose_year(profile_type: str, soph: float, strategy: Optional[BotStrategy]) -> int:
    y = now_year()
    if profile_type == "bot_farm":
        if strategy and strategy.low_and_slow > 0.7:
            w_recent, w_mid, w_old = 0.60, 0.32, 0.08
        elif strategy and strategy.name == "low_cost_loop":
            w_recent, w_mid, w_old = 0.84, 0.14, 0.02
        else:
            w_recent, w_mid, w_old = 0.74, 0.22, 0.04
    elif profile_type == "hacked_account":
        w_recent, w_mid, w_old = 0.66, 0.28, 0.06
    else:
        w_recent, w_mid, w_old = 0.60, 0.32, 0.08

    comp = weighted_choice(["recent", "mid", "old"], [w_recent, w_mid, w_old])

    if comp == "recent":
        back = int(np.random.choice([0, 1, 2, 3, 4], p=[0.36, 0.22, 0.17, 0.15, 0.10]))
        year = y - back
    elif comp == "mid":
        center = 2012 - int(3.0 * (1.0 - soph))
        year = int(np.random.normal(loc=center, scale=5.8))
    else:
        back = int(np.random.exponential(scale=9.0 + 4.0 * (1.0 - soph)))
        year = y - (10 + back)

    return clamp_int(year, EARLIEST_YEAR, y)

# ---------------------------- Sessions ---------------------------- #

@dataclass
class SessionContext:
    device_hash: str
    ip_block_24: str
    user_agent: str
    compromised: bool


def sample_sessions_for_account(profile_type: str, policy: PipelinePolicy, upload_count: int, strategy: Optional[BotStrategy]) -> int:
    burst = policy.session_burstiness
    expected_len = 2.5 + 7.0 * burst
    if profile_type == "bot_farm":
        expected_len *= 1.18
        if strategy and strategy.volume_profile == "low":
            expected_len *= 0.70
        if strategy and strategy.low_and_slow > 0.7:
            expected_len *= 0.85
    n = int(max(1, round(upload_count / expected_len + np.random.uniform(-0.3, 0.7))))
    return clamp_int(n, 1, max(1, min(55, upload_count)))


def allocate_session_lengths(upload_count: int, n_sessions: int, burstiness: float, low_and_slow: float = 0.0) -> List[int]:
    p = 0.40 - 0.25 * burstiness + 0.10 * low_and_slow
    p = max(0.14, min(0.60, p))

    lengths = [max(1, int(np.random.geometric(p))) for _ in range(n_sessions)]
    total = sum(lengths) or 1
    scaled = [max(1, int(round(l * upload_count / total))) for l in lengths]

    diff = upload_count - sum(scaled)
    i = 0
    while diff != 0 and i < 300000:
        j = i % len(scaled)
        if diff > 0:
            scaled[j] += 1
            diff -= 1
        else:
            if scaled[j] > 1:
                scaled[j] -= 1
                diff += 1
        i += 1
    return scaled


def choose_compromised_session_flags(n_sessions: int, account: "Account") -> List[bool]:
    if account.profile_type != "hacked_account":
        return [False] * n_sessions
    k = min(n_sessions, max(1, account.compromised_sessions))
    flags = [False] * n_sessions
    idxs = sorted(random.sample(range(n_sessions), k=k))
    for idx in idxs:
        flags[idx] = True
        if random.random() < 0.45 and idx - 1 >= 0:
            flags[idx - 1] = True
        if random.random() < 0.45 and idx + 1 < n_sessions:
            flags[idx + 1] = True
    return flags

# ---------------------------- Titles / releases ---------------------------- #

@dataclass
class Release:
    album_name: str
    year: int
    genre: str
    track_total: int
    quality_bias: float
    durations: List[int]


def choose_genre(genre_probs: np.ndarray) -> str:
    return str(np.random.choice(GENRES, p=genre_probs))


def make_album_name(fake: Faker, genre: str, strictness: float, profile_type: str, strategy: Optional[BotStrategy]) -> str:
    tokens = tokens_for_genre(genre)
    k = int(np.random.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
    base = title_case(random.sample(tokens, k=k))
    if random.random() < (0.10 * (1.0 - strictness)):
        base = f"{base} Vol. {random.randint(1, 99)}"
    if profile_type == "bot_farm" and strategy and strategy.name == "low_cost_loop" and random.random() < 0.14:
        base = f"{base} Collection {random.randint(1, 60)}"
    if random.random() < (0.08 + 0.06 * strictness):
        base = base + random.choice([" EP", " (Deluxe)", " (Expanded)"])
    return base.strip()


def make_track_title(fake: Faker, genre: str, strictness: float, profile_type: str,
                     album_track: bool, track_no: Optional[int], track_total: Optional[int],
                     strategy: Optional[BotStrategy]) -> str:
    tokens = tokens_for_genre(genre)
    templated = (profile_type == "bot_farm" and strategy and strategy.name == "low_cost_loop" and random.random() < 0.16)
    if templated:
        base = f"Track {random.randint(1, 999):03d}"
    else:
        if random.random() < strictness:
            n_words = int(np.random.choice([2, 3], p=[0.45, 0.55]))
        else:
            n_words = int(np.random.choice([1, 2, 3, 4], p=[0.10, 0.40, 0.35, 0.15]))
        base = title_case(random.sample(tokens + (TOKENS_COMMON if genre != "Pop" else []), k=n_words))

    if random.random() < (0.09 * strictness) and profile_type != "bot_farm":
        base += f" ({random.choice(['Remastered', 'Edit', 'Live', 'Acoustic'])})"
    if random.random() < (0.10 if genre in ("Pop", "Hip-Hop") else 0.04) and profile_type != "bot_farm":
        base += f" (feat. {fake.first_name()} {fake.last_name()})"

    if album_track and track_no and track_total:
        style = weighted_choice(["none", "01", "1.", "01/12"], [0.50, 0.25, 0.16, 0.09])
        if style == "01":
            return f"{track_no:02d} {base}"
        if style == "1.":
            return f"{track_no}. {base}"
        if style == "01/12":
            return f"{track_no:02d}/{track_total:02d} {base}"
    return base


def album_duration_profile(genre: str, profile_type: str, strategy: Optional[BotStrategy]) -> Tuple[float, float, int, int]:
    if genre == "Pop":
        mu, sd, lo, hi = 205, 28, 55, 480
    elif genre == "Hip-Hop":
        mu, sd, lo, hi = 195, 32, 55, 540
    elif genre == "Ambient":
        mu, sd, lo, hi = 390, 95, 90, 3000
    else:
        mu, sd, lo, hi = 175, 80, 35, 2400

    if profile_type == "bot_farm" and strategy:
        if strategy.name == "low_cost_loop":
            mu *= 0.72
            sd *= 0.70
        elif strategy.name == "sharded_variants":
            sd *= 0.85
        if strategy.low_and_slow > 0.7:
            sd *= 0.85

    return mu, sd, lo, hi


def correlated_album_durations(genre: str, profile_type: str, strategy: Optional[BotStrategy], track_total: int) -> List[int]:
    mu, sd, lo, hi = album_duration_profile(genre, profile_type, strategy)
    mu = float(mu + np.random.normal(0, sd * 0.25))

    durations: List[int] = []
    eps = float(np.random.normal(0, sd))
    for _ in range(track_total):
        eps = 0.65 * eps + float(np.random.normal(0, sd * 0.55))
        d = truncated_int_sampler(lambda: mu + eps, lo, hi)
        durations.append(int(d))

    if random.random() < 0.10 and track_total >= 6 and profile_type != "bot_farm":
        idx = random.randint(0, min(2, track_total - 1))
        durations[idx] = truncated_int_sampler(lambda: np.random.normal(65, 18), 25, 150)

    return durations


def plan_releases(fake: Faker, account: Account, upload_count: int) -> List[Tuple[Optional[Release], Optional[int]]]:
    pt = account.profile_type
    strict = account.policy.metadata_strictness
    strategy = account.bot_strategy

    if pt == "bot_farm" and strategy:
        if strategy.name == "low_cost_loop":
            p_album = 0.18
        elif strategy.name in ("label_mimic", "compliance_mimic"):
            p_album = 0.50
        elif strategy.name == "low_and_slow":
            p_album = 0.34
        else:
            p_album = 0.30
        p_album *= (0.85 + 0.25 * (1.0 - strategy.low_and_slow))
    elif pt == "hacked_account":
        p_album = 0.34
    else:
        p_album = 0.44 + 0.12 * account.sophistication

    max_albums = max(0, min(9, upload_count // 4))
    albums_count = 0
    if max_albums > 0 and random.random() < p_album:
        albums_count = int(np.random.choice([1, 2, 3, 4, 5], p=[0.52, 0.26, 0.12, 0.06, 0.04]))
        albums_count = min(albums_count, max_albums)

    releases: List[Release] = []
    plan: List[Tuple[Optional[Release], Optional[int]]] = []

    for _ in range(albums_count):
        genre = choose_genre(account.genre_probs)
        year = choose_year(pt, account.sophistication, strategy)

        if genre in ("Ambient", "Noise"):
            base_tracks = int(np.random.choice([6, 8, 10, 12, 14, 16, 18], p=[0.08, 0.15, 0.20, 0.22, 0.18, 0.10, 0.07]))
        else:
            base_tracks = int(np.random.choice([4, 5, 6, 8, 10, 12, 14, 16], p=[0.10, 0.16, 0.18, 0.20, 0.16, 0.10, 0.06, 0.04]))

        if pt == "bot_farm" and strategy and strategy.name == "low_cost_loop":
            base_tracks = clamp_int(base_tracks + int(np.random.choice([-2, -1, 0, 2], p=[0.12, 0.36, 0.36, 0.16])), 3, 16)

        q_bias = float(np.clip(
            0.18 + 0.68 * account.sophistication + 0.14 * strict + np.random.normal(0, 0.09),
            0.0, 1.0
        ))
        if pt == "bot_farm" and strategy:
            q_bias = float(np.clip(q_bias + 0.12 * strategy.compliance_mimic - 0.06 * (1.0 - strategy.compliance_mimic), 0.0, 1.0))

        album_name = make_album_name(fake, genre, strict, pt, strategy)
        durations = correlated_album_durations(genre, pt, strategy, base_tracks)
        releases.append(Release(album_name, year, genre, base_tracks, q_bias, durations))

    remaining = upload_count
    if releases:
        if pt == "bot_farm":
            frac_album = np.random.beta(2.0, 2.7)
            if strategy and strategy.low_and_slow > 0.7:
                frac_album = min(0.72, frac_album)
        elif pt == "hacked_account":
            frac_album = np.random.beta(2.0, 3.0)
        else:
            frac_album = np.random.beta(2.4, 2.1)

        n_album_uploads = clamp_int(frac_album * upload_count, 0, upload_count)
        remaining = upload_count - n_album_uploads

        for rel in releases:
            if n_album_uploads <= 0:
                break
            take = min(n_album_uploads, rel.track_total)
            if random.random() < 0.35:
                take = max(1, int(round(take * np.random.uniform(0.55, 0.92))))
            take = min(take, n_album_uploads)
            track_nos = sorted(random.sample(range(1, rel.track_total + 1), k=take))
            for tno in track_nos:
                plan.append((rel, tno))
            n_album_uploads -= take

    for _ in range(remaining):
        plan.append((None, None))

    random.shuffle(plan)
    return plan

# ---------------------------- Collaborators ---------------------------- #

def build_collaborator_pool(fake: Faker) -> List[str]:
    size = int(np.random.choice([0, 1, 2, 3, 4, 5], p=[0.25, 0.22, 0.18, 0.16, 0.12, 0.07]))
    pool: List[str] = []
    for _ in range(size):
        pool.append(f"{fake.first_name()} {fake.last_name()}" if random.random() < 0.60 else fake.user_name().replace("_", " ").title())
    out: List[str] = []
    seen = set()
    for n in pool:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def sample_collaborators(fake: Faker, genre: str, strictness: float, profile_type: str,
                         collaborator_pool: List[str], strategy: Optional[BotStrategy]) -> str:
    if profile_type == "bot_farm":
        p_any = 0.04 + (0.08 if strategy and strategy.compliance_mimic > 0.7 else 0.02)
    else:
        base = 0.10 if genre in ("Ambient", "Noise") else 0.18
        p_any = base + 0.18 * strictness

    if random.random() > p_any:
        return json_list_str([])

    k = int(np.random.choice([1, 2, 3], p=[0.68, 0.24, 0.08]))
    picks = random.sample(collaborator_pool, k=min(k, len(collaborator_pool))) if collaborator_pool else []
    if not picks:
        picks = [f"{fake.first_name()} {fake.last_name()}"]

    out: List[str] = []
    seen = set()
    for n in picks:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return json_list_str(out)

# ---------------------------- Durations (singles) ---------------------------- #

def single_duration(genre: str, profile_type: str, strategy: Optional[BotStrategy], compromised: bool) -> int:
    if genre == "Pop":
        mu, sd, lo, hi = 210, 42, 55, 480
    elif genre == "Hip-Hop":
        mu, sd, lo, hi = 200, 52, 55, 540
    elif genre == "Ambient":
        mu, sd, lo, hi = 365, 135, 90, 3000
    else:
        mu, sd, lo, hi = 175, 95, 35, 2400

    if compromised:
        mu *= np.random.uniform(0.82, 1.00)
        sd *= np.random.uniform(0.90, 1.20)

    if profile_type == "bot_farm" and strategy:
        if strategy.name == "low_cost_loop":
            r = random.random()
            if r < 0.55:
                return truncated_int_sampler(lambda: np.random.gamma(shape=5.0, scale=8.0) + 20.0, 25, 120)
            if r < 0.90:
                return truncated_int_sampler(lambda: np.random.normal(mu * 0.95, sd * 0.85), 90, min(650, hi))
            return truncated_int_sampler(lambda: np.random.normal(520, 150), 220, min(1600, hi))
        if strategy.name == "sharded_variants":
            return truncated_int_sampler(lambda: np.random.normal(mu * 0.92, sd * 0.95), lo, hi)
        return truncated_int_sampler(lambda: np.random.normal(mu, sd * (0.85 if strategy.low_and_slow > 0.6 else 0.95)), lo, hi)

    if profile_type != "bot_farm" and random.random() < 0.025:
        return truncated_int_sampler(lambda: np.random.normal(70, 18), 25, 170)

    return truncated_int_sampler(lambda: np.random.normal(mu, sd), lo, hi)


def single_album_field(fake: Faker, genre: str, strict: float, profile_type: str, strategy: Optional[BotStrategy]) -> str:
    if random.random() < (0.60 + 0.10 * strict):
        return "Single"
    return make_album_name(fake, genre, strict, profile_type, strategy)

# ---------------------------- Fingerprints ---------------------------- #

@dataclass
class FingerprintContext:
    local_assets: List[str]
    local_families: List[str]


def sample_fingerprints(profile_type: str, farm: Optional[Farm], strategy: Optional[BotStrategy],
                        fp_ctx: FingerprintContext, compromised: bool) -> Tuple[str, str]:
    if profile_type == "normal_user":
        p_asset, p_fam = 0.02, 0.03
    elif profile_type == "hacked_account":
        p_asset, p_fam = 0.07, 0.10
    else:
        p_asset, p_fam = 0.16, 0.22

    if compromised:
        p_asset = min(0.35, p_asset + 0.10)
        p_fam = min(0.40, p_fam + 0.12)

    if profile_type == "bot_farm" and strategy:
        if strategy.reuse_mode == "high_exact":
            p_asset = min(0.58, p_asset + 0.22)
            p_fam = min(0.52, p_fam + 0.08)
        elif strategy.reuse_mode == "high_phash":
            p_asset = max(0.10, p_asset - 0.07)
            p_fam = min(0.58, p_fam + 0.18)
        elif strategy.reuse_mode == "low_reuse":
            p_asset = max(0.05, p_asset - 0.10)
            p_fam = max(0.06, p_fam - 0.12)

        if strategy.low_and_slow > 0.7:
            p_asset *= 0.55
            p_fam *= 0.65

    if random.random() < p_asset:
        if profile_type == "bot_farm" and farm:
            asset_id = random.choice(farm.asset_pool)
        elif fp_ctx.local_assets:
            asset_id = random.choice(fp_ctx.local_assets)
        else:
            asset_id = h_sha1("asset::new::" + uuid.uuid4().hex)
    else:
        asset_id = h_sha1("asset::new::" + uuid.uuid4().hex)
        fp_ctx.local_assets.append(asset_id)
        if len(fp_ctx.local_assets) > 1500:
            fp_ctx.local_assets = fp_ctx.local_assets[-950:]

    if random.random() < p_fam:
        if profile_type == "bot_farm" and farm:
            fam_id = random.choice(farm.phash_families)
        elif fp_ctx.local_families:
            fam_id = random.choice(fp_ctx.local_families)
        else:
            fam_id = h_sha1("phfam::new::" + uuid.uuid4().hex)
    else:
        fam_id = h_sha1("phfam::new::" + uuid.uuid4().hex)
        fp_ctx.local_families.append(fam_id)
        if len(fp_ctx.local_families) > 1500:
            fp_ctx.local_families = fp_ctx.local_families[-950:]

    return h_sha1("AUDIO::" + asset_id), h_md5("PHASH::" + fam_id)

# ---------------------------- Quality (TUNED BITRATE) ---------------------------- #

def choose_format_bitrate(profile_type: str, policy: PipelinePolicy, lossless_required: bool,
                          soph: float, year: int, genre: str, album_track: bool,
                          release_quality_bias: float, strategy: Optional[BotStrategy],
                          compromised: bool) -> Tuple[str, int]:
    y_now = now_year()
    recency = max(0.0, min(1.0, 1.0 - (y_now - year) / 12.0))
    age_years = max(0, y_now - year)
    legacy_bias = float(np.clip((age_years - 8) / 20.0, 0.0, 1.0))  # tuned stronger & earlier onset

    q = 0.38 * soph + 0.24 * recency + 0.20 * release_quality_bias + 0.18 * policy.lossless_preference
    if genre == "Ambient":
        q += 0.03
    if compromised:
        q -= 0.05

    smartness = 0.0
    if profile_type == "bot_farm" and strategy:
        q += 0.22 * strategy.compliance_mimic
        q -= 0.12 * (1.0 - strategy.compliance_mimic)
        q -= 0.16 if strategy.name == "low_cost_loop" else 0.0
        q -= 0.10 if strategy.name == "sharded_variants" else 0.0
        smartness = float(np.clip(0.35 * strategy.compliance_mimic + 0.65 * strategy.low_and_slow, 0.0, 1.0))
        if random.random() < (0.10 if strategy.compliance_mimic > 0.7 else 0.06):
            q = min(1.0, q + 0.18)
    q = float(np.clip(q, 0.0, 1.0))

    enforce_lossless = (lossless_required and album_track and random.random() < 0.96)

    # --- Format probabilities (tuned to allow more legacy MP3) ---
    if profile_type != "bot_farm":
        p_mp3 = 0.03 + 0.14 * legacy_bias + 0.04 * (1.0 - policy.lossless_preference)
        p_mp3 = float(np.clip(p_mp3, 0.01, 0.28))
        p_flac = 0.54 + 0.28 * q + 0.12 * policy.lossless_preference - 0.14 * legacy_bias
        p_wav = 1.0 - p_mp3 - p_flac
        probs = np.array([p_mp3, max(0.02, p_wav), max(0.05, p_flac)], dtype=float)
    else:
        if strategy and strategy.name == "low_cost_loop":
            p_mp3 = 0.52 - 0.22 * q + 0.10 * legacy_bias
        elif strategy and strategy.compliance_mimic > 0.8:
            p_mp3 = 0.06 + 0.08 * legacy_bias - 0.03 * q
        else:
            p_mp3 = 0.22 + 0.10 * legacy_bias - 0.08 * q
        p_mp3 = float(np.clip(p_mp3, 0.02, 0.80))

        p_flac = 0.40 + 0.34 * q + 0.10 * (strategy.compliance_mimic if strategy else 0.0) - 0.12 * legacy_bias
        p_wav = 1.0 - p_mp3 - p_flac
        probs = np.array([p_mp3, max(0.02, p_wav), max(0.05, p_flac)], dtype=float)

    if enforce_lossless:
        probs[0] = 0.01
        probs[2] += 0.18
        probs[1] -= 0.10

    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()
    fmt = str(np.random.choice(["mp3", "wav", "flac"], p=probs))

    # --- Bitrate tiers (TUNED so lossless isn't always 320) ---
    def lossless_bitrate_tier() -> int:
        # reporting quirks: even lossless masters sometimes show up as 256-tier in metadata exports
        reporting_quirk = (random.random() < (0.10 + 0.08 * legacy_bias))  # more common for older/legacy
        # Base distribution favors 320 but not overwhelmingly; q increases 320, legacy increases 256/192.
        p128 = 0.01
        p192 = 0.08 + 0.10 * legacy_bias + 0.05 * (1.0 - q)
        p256 = 0.32 + 0.22 * legacy_bias + 0.10 * (1.0 - q)
        p320 = 1.0 - (p128 + p192 + p256)
        p = np.array([p128, p192, p256, max(0.05, p320)], dtype=float)
        p = np.clip(p, 0.01, None)
        p = p / p.sum()
        br = int(np.random.choice(BITRATES, p=p))

        if reporting_quirk and br == 320 and random.random() < 0.85:
            br = 256
        if br == 128 and random.random() < 0.96:
            br = int(np.random.choice([256, 320], p=[0.55, 0.45]))
        return br

    def mp3_bitrate_tier() -> int:
        # Tuned: reduce "always 320", especially for newer uploads; allow more 192/256.
        base_new = np.array([0.06, 0.18, 0.36, 0.40])  # was more 320-heavy
        base_old = np.array([0.52, 0.30, 0.14, 0.04])
        p = (1.0 - legacy_bias) * base_new + legacy_bias * base_old

        # quality propensity pushes upward
        p_q = np.array([0.10, 0.20, 0.35, 0.35]) * (1.0 - q) + np.array([0.05, 0.10, 0.38, 0.47]) * q
        p = 0.55 * p + 0.45 * p_q

        # low_cost_loop pulls down tiers
        if profile_type == "bot_farm" and strategy and strategy.name == "low_cost_loop" and random.random() < 0.75:
            p = 0.70 * p + 0.30 * np.array([0.55, 0.25, 0.14, 0.06])

        # smart bots diversify (avoid over-peaking at 320)
        if profile_type == "bot_farm" and smartness > 0.6 and random.random() < 0.70:
            p = 0.85 * p + 0.15 * np.array([0.06, 0.16, 0.42, 0.36])

        p = np.clip(p, 0.01, None)
        p = p / p.sum()
        return int(np.random.choice(BITRATES, p=p))

    if fmt in ("wav", "flac"):
        br = lossless_bitrate_tier()
    else:
        br = mp3_bitrate_tier()

    return fmt, int(br)

# ---------------------------- IP / device selection ---------------------------- #

def pick_ip_block_for_session(fake: Faker, account: Account, farm: Optional[Farm], compromised: bool) -> str:
    if account.profile_type == "bot_farm" and farm and account.bot_strategy:
        strat = account.bot_strategy
        if strat.infra_style == "residential_proxy":
            if account.bot_home_blocks_24 and random.random() < 0.82:
                return random.choice(account.bot_home_blocks_24)
            if random.random() < 0.10 and farm.proxy_blocks_24:
                return random.choice(farm.proxy_blocks_24)
            if random.random() < 0.06:
                return random.choice(farm.datacenter_blocks_24)
            return ipv4_block_24(fake)
        r = random.random()
        if r < 0.78:
            return random.choice(farm.datacenter_blocks_24)
        if r < 0.92 and farm.proxy_blocks_24:
            return random.choice(farm.proxy_blocks_24)
        return ipv4_block_24(fake)

    if account.profile_type == "hacked_account":
        if compromised:
            r = random.random()
            if r < 0.55 and farm and farm.proxy_blocks_24:
                return random.choice(farm.proxy_blocks_24)
            if r < 0.80 and farm:
                return random.choice(farm.datacenter_blocks_24)
            return ipv4_block_24(fake)
        if account.home_blocks_24 and random.random() < 0.78:
            return random.choice(account.home_blocks_24)
        return ipv4_block_24(fake)

    if account.home_blocks_24 and random.random() < 0.84:
        return random.choice(account.home_blocks_24)
    return ipv4_block_24(fake)


def pick_device_for_session(account: Account, compromised: bool) -> str:
    if account.profile_type == "hacked_account" and compromised and random.random() < 0.35:
        return h_md5(f"compromised::{account.account_external_id}::{uuid.uuid4().hex}")[:16]
    return random.choice(account.devices).device_hash

# ---------------------------- Generate accounts ---------------------------- #

def generate_accounts(fake: Faker, farms: Dict[int, Farm], target_rows: int) -> Tuple[List[Account], List[int]]:
    accounts: List[Account] = []
    counts: List[int] = []
    rows_so_far = 0
    farm_ids = list(farms.keys())

    while rows_so_far < target_rows * 1.03:
        profile_type = sample_profile_type()
        expected_category = expected_category_from_profile(profile_type)

        strategy: Optional[BotStrategy] = None
        farm_id: Optional[int] = None
        bot_home: List[str] = []

        if profile_type == "bot_farm":
            farm_id = random.choice(farm_ids)
            strategy = farms[farm_id].strategy

        soph = sample_sophistication(profile_type)
        account_type = sample_account_type(soph, profile_type)
        pipeline = sample_pipeline(profile_type, account_type, soph, strategy)
        policy = PIPELINE_POLICIES[pipeline]
        lossless_required = sample_lossless_required(policy, soph, account_type, profile_type, strategy)
        genre_probs = sample_genre_probs(profile_type, strategy)

        account_id = uuid_hex("acc_")
        display_name = make_display_name(fake, profile_type)

        if profile_type == "bot_farm" and strategy:
            smartness = float(np.clip(0.35 * strategy.compliance_mimic + 0.65 * strategy.low_and_slow + np.random.normal(0, 0.06), 0.0, 1.0))
        else:
            smartness = float(np.clip(0.20 * soph + np.random.normal(0, 0.05), 0.0, 1.0))

        home_blocks = sample_home_blocks(fake, profile_type)
        n_devices = sample_num_devices(profile_type, policy, strategy)

        if profile_type == "bot_farm" and farm_id is not None and strategy:
            farm = farms[farm_id]
            if strategy.infra_style == "residential_proxy" and farm.residential_proxy_blocks_24:
                k = int(np.random.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
                bot_home = random.sample(farm.residential_proxy_blocks_24, k=min(k, len(farm.residential_proxy_blocks_24)))

        if profile_type == "bot_farm" and farm_id is not None:
            farm = farms[farm_id]
            devices: List[Device] = []
            for _ in range(n_devices):
                if strategy and smartness > 0.7 and random.random() < 0.70:
                    dev_hash = h_md5(f"bot_unique::{account_id}::{uuid.uuid4().hex}")[:16]
                else:
                    dev_hash = random.choice(farm.device_pool)
                devices.append(Device(dev_hash, bot_home[:]))
        else:
            devices = build_devices(account_id, home_blocks, n_devices)

        compromised_fraction = 0.0
        compromised_sessions = 0
        if profile_type == "hacked_account":
            compromised_fraction = float(np.clip(np.random.beta(2.2, 5.0) * 0.9, 0.05, 0.55))
            compromised_sessions = int(np.random.choice([1, 1, 2, 2, 3], p=[0.32, 0.18, 0.24, 0.18, 0.08]))

        n_uploads = sample_upload_count(profile_type, strategy)

        accounts.append(Account(
            account_external_id=account_id,
            profile_type=profile_type,
            expected_category=expected_category,
            account_type=account_type,
            display_name=display_name,
            pipeline=pipeline,
            policy=policy,
            lossless_required=lossless_required,
            sophistication=soph,
            smartness=smartness,
            genre_probs=genre_probs,
            home_blocks_24=home_blocks,
            devices=devices,
            compromised_fraction=compromised_fraction,
            compromised_sessions=compromised_sessions,
            farm_id=farm_id,
            bot_strategy=strategy,
            bot_home_blocks_24=bot_home,
        ))
        counts.append(n_uploads)
        rows_so_far += n_uploads

    return accounts, counts

# ---------------------------- Main writer ---------------------------- #

def generate_and_write_csv(fake: Faker, accounts: List[Account], counts: List[int], farms: Dict[int, Farm], target_rows: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    header_written = False
    rows_written = 0

    label_counts = {"legit": 0, "fraud": 0}
    profile_counts = {p: 0 for p in PROFILE_TYPES}
    unique_accounts = set()
    sample_rows: List[Dict[str, object]] = []

    seen_upload_ids = set()
    seen_content_refs = set()

    for account, upload_count in zip(accounts, counts):
        if rows_written >= target_rows:
            break

        farm = farms.get(account.farm_id) if account.farm_id else None
        strict = account.policy.metadata_strictness
        collab_pool = build_collaborator_pool(fake)
        plan = plan_releases(fake, account, upload_count)

        n_sessions = sample_sessions_for_account(account.profile_type, account.policy, upload_count, account.bot_strategy)
        low_and_slow = account.bot_strategy.low_and_slow if (account.profile_type == "bot_farm" and account.bot_strategy) else 0.0
        sess_lengths = allocate_session_lengths(upload_count, n_sessions, account.policy.session_burstiness, low_and_slow=low_and_slow)
        compromised_flags = choose_compromised_session_flags(n_sessions, account)

        fp_ctx = FingerprintContext([], [])
        sessions: List[SessionContext] = []

        for s_idx in range(n_sessions):
            comp = compromised_flags[s_idx]
            dev = pick_device_for_session(account, comp)
            block = pick_ip_block_for_session(fake, account, farm, comp)
            ua = sample_user_agent(account.profile_type, account.policy, smartness=account.smartness, compromised=comp)
            sessions.append(SessionContext(dev, block, ua, comp))

        chunk_rows: List[Dict[str, object]] = []
        plan_idx = 0

        for s_idx, slen in enumerate(sess_lengths):
            if rows_written >= target_rows:
                break

            sess = sessions[s_idx]
            compromised = sess.compromised

            if account.profile_type == "hacked_account":
                if compromised and random.random() < max(0.0, 1.0 - account.compromised_fraction):
                    compromised = random.random() < 0.45

            for _ in range(slen):
                if rows_written >= target_rows or plan_idx >= len(plan):
                    break

                rel, track_no = plan[plan_idx]
                plan_idx += 1

                upload_external_id = uuid_hex("upl_")
                content_ref = str(uuid.uuid4())

                while upload_external_id in seen_upload_ids:
                    upload_external_id = uuid_hex("upl_")
                while content_ref in seen_content_refs:
                    content_ref = str(uuid.uuid4())

                seen_upload_ids.add(upload_external_id)
                seen_content_refs.add(content_ref)

                if rel is not None and track_no is not None:
                    genre = rel.genre
                    year = rel.year
                    album_name = rel.album_name
                    album_track = True
                    release_q = rel.quality_bias
                    duration = int(rel.durations[track_no - 1])
                    title = make_track_title(fake, genre, strict, account.profile_type, True, track_no, rel.track_total, account.bot_strategy)
                else:
                    genre = choose_genre(account.genre_probs)
                    year = choose_year(account.profile_type, account.sophistication, account.bot_strategy)
                    album_name = single_album_field(fake, genre, strict, account.profile_type, account.bot_strategy)
                    album_track = False
                    release_q = float(np.clip(0.10 + 0.65 * account.sophistication + 0.10 * strict + np.random.normal(0, 0.10), 0, 1))
                    duration = single_duration(genre, account.profile_type, account.bot_strategy, compromised)
                    title = make_track_title(fake, genre, strict, account.profile_type, False, None, None, account.bot_strategy)

                fmt, br = choose_format_bitrate(
                    account.profile_type, account.policy, account.lossless_required,
                    account.sophistication, year, genre, album_track, release_q,
                    account.bot_strategy, compromised
                )

                collaborators = sample_collaborators(fake, genre, strict, account.profile_type, collab_pool, account.bot_strategy)
                audio_hash, phash = sample_fingerprints(account.profile_type, farm, account.bot_strategy, fp_ctx, compromised)

                device_hash = sess.device_hash
                ip = ip_from_block(sess.ip_block_24)
                ua = sess.user_agent
                if account.profile_type == "hacked_account" and compromised and random.random() < 0.12:
                    ua = sample_user_agent(account.profile_type, account.policy, smartness=account.smartness, compromised=True)

                row = {
                    "account_external_id": account.account_external_id,
                    "account_type": account.account_type,
                    "display_name": account.display_name,
                    "upload_external_id": upload_external_id,
                    "metadata_title": title,
                    "metadata_genre": genre,
                    "metadata_duration_seconds": int(duration),
                    "metadata_bitrate": int(br),
                    "metadata_format": fmt,
                    "metadata_collaborators": collaborators,
                    "metadata_album": album_name,
                    "metadata_year": int(year),
                    "content_ref": content_ref,
                    "fingerprints_audio_hash": audio_hash,
                    "fingerprints_perceptual_hash": phash,
                    "device_context_device_hash": device_hash,
                    "device_context_ip": ip,
                    "device_context_user_agent": ua,
                    "expected_category": account.expected_category,
                    "profile_type": account.profile_type,
                }

                label_counts[account.expected_category] += 1
                profile_counts[account.profile_type] += 1
                unique_accounts.add(account.account_external_id)
                if len(sample_rows) < 5:
                    sample_rows.append(row)

                chunk_rows.append(row)
                rows_written += 1

                if len(chunk_rows) >= CHUNK_SIZE:
                    df_chunk = pd.DataFrame(chunk_rows, columns=COLUMNS)
                    df_chunk.to_csv(OUT_PATH, index=False, mode="a", header=(not header_written), encoding="utf-8")
                    header_written = True
                    chunk_rows = []

        if chunk_rows:
            df_chunk = pd.DataFrame(chunk_rows, columns=COLUMNS)
            df_chunk.to_csv(OUT_PATH, index=False, mode="a", header=(not header_written), encoding="utf-8")
            header_written = True

    assert rows_written == target_rows, f"Expected {target_rows} rows, got {rows_written}"
    assert len(seen_upload_ids) == target_rows, "upload_external_id uniqueness violated"
    assert len(seen_content_refs) == target_rows, "content_ref uniqueness violated"
    assert len(unique_accounts) < target_rows, "Expected multiple uploads per account"

    print(f"Saved {rows_written} rows to {OUT_PATH}")
    print("\nexpected_category distribution:")
    print(pd.Series(label_counts))
    print("\nprofile_type distribution:")
    print(pd.Series(profile_counts))
    print(f"\nUnique accounts: {len(unique_accounts)}")
    print("\nExample rows:")
    if sample_rows:
        print(pd.DataFrame(sample_rows, columns=COLUMNS).to_string(index=False))

# ---------------------------- Main ---------------------------- #

def main() -> None:
    seed_everything(SEED)
    fake = Faker()

    farms = build_farms(fake, n_rows=N_ROWS)
    accounts, counts = generate_accounts(fake, farms, target_rows=N_ROWS)

    generate_and_write_csv(fake, accounts, counts, farms, target_rows=N_ROWS)

if __name__ == "__main__":
    main()
