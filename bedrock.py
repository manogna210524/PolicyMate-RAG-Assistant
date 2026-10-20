"""All calls to AWS Bedrock: embeddings (Titan) and text generation (Converse API)."""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import boto3
import numpy as np
from botocore.config import Config

from core import config


@lru_cache(maxsize=1)
def get_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            read_timeout=60,
        ),
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed many texts in parallel. Returns a float32 array of shape (n, dims)."""
    client = get_client()

    def _embed_one(text: str) -> list[float]:
        body = json.dumps({
            "inputText": text,
            "dimensions": config.EMBED_DIMENSIONS,
            "normalize": True,
        })
        resp = client.invoke_model(
            modelId=config.EMBED_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(resp["body"].read())["embedding"]

    with ThreadPoolExecutor(max_workers=config.EMBED_WORKERS) as pool:
        vectors = list(pool.map(_embed_one, texts))
    return np.asarray(vectors, dtype="float32")


def generate_text(system_prompt: str, user_prompt: str) -> str:
    """One Converse call. Returns the model's text reply."""
    resp = get_client().converse(
        modelId=config.GEN_MODEL_ID,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        inferenceConfig={
            "maxTokens": config.MAX_ANSWER_TOKENS,
            "temperature": config.TEMPERATURE,
        },
    )
    parts = resp["output"]["message"]["content"]
    return "".join(p.get("text", "") for p in parts).strip()
