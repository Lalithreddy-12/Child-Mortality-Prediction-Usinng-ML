# hf_test.py
import os
import json
import traceback
from dotenv import load_dotenv

# load .env in current folder
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
print("HF_API_KEY present?:", bool(HF_API_KEY))
if HF_API_KEY:
    print("HF_API_KEY prefix (masked):", HF_API_KEY[:8] + "...")

# Use huggingface_hub InferenceClient if available
try:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=HF_API_KEY, timeout=30)
    model_name = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    print("Using model:", model_name)
    try:
        resp = client.chat_completion(model=model_name,
                                     messages=[{"role": "user", "content": "what is the capital of India?"}],
                                     max_tokens=50)
        # best-effort print
        try:
            content = resp.choices[0].message["content"]
            print("HF response excerpt:", content[:1000])
        except Exception:
            # fallback: print full obj repr truncated
            txt = repr(resp)
            print("HF response repr (truncated):", txt[:1000])
    except Exception as e:
        print("Error calling HF chat_completion:")
        print(type(e), e)
        print(traceback.format_exc()[:2000])
except Exception as e:
    print("huggingface_hub import failed or client error:", e)
    print("traceback (truncated):")
    print(traceback.format_exc()[:2000])
