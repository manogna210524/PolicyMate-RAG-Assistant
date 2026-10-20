"""Settings: paths, model IDs, and retrieval tuning. Change them here only."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
POLICIES_CSV = ROOT / "data" / "policies.csv"

# One folder per policy: storage/<id>/index.faiss (vectors) and storage/<id>/chunks.json (text)
STORAGE_DIR = ROOT / "storage"

# AWS Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", "us.amazon.nova-pro-v1:0")
EMBED_DIMENSIONS = 1024
EMBED_WORKERS = 8
MAX_ANSWER_TOKENS = 700
TEMPERATURE = 0.1

# Chunking and retrieval
CHUNK_SIZE = 2000      # characters
CHUNK_OVERLAP = 200    # characters shared by neighbouring chunks on the same page
TOP_K = 5              # chunks sent to the model
# Cosine similarity (vectors are normalized). Below this, the policy is treated as
# not covering the question. Tune it with real questions after the first build.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.30"))
