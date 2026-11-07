import time
from google import genai
from google.genai import errors
from config import GOOGLE_API_KEY, GEMINI_MODEL

_client = None

# 429s that won't clear by waiting (out of credits, daily quota). Per-minute 429s are retried.
QUOTA_MARKERS = ("credits", "depleted", "prepay", "per day", "perday")


class LLMQuotaError(Exception):
    """Raised when the Gemini account is out of credits/quota, so callers can stop early."""


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def generate(prompt, delay=4.1, retries=1):
    """Single-turn Gemini call. Returns the response text, or None on a non-fatal error."""
    for attempt in range(retries + 1):
        try:
            response = get_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
            time.sleep(delay)
            return (response.text or "").strip() or None
        except errors.APIError as e:
            message = str(e).lower()
            # 402 = payment required (credits depleted)
            if e.code == 402 or (e.code == 429 and any(m in message for m in QUOTA_MARKERS)):
                raise LLMQuotaError(str(e)) from e
            print(f"  Gemini error ({e.code}), attempt {attempt + 1}: {str(e)[:200]}")
            # Per-minute rate limit: back off before retrying
            time.sleep(30 if e.code == 429 else delay)
        except Exception as e:
            print(f"  Gemini error, attempt {attempt + 1}: {e}")
            time.sleep(delay)
    return None
