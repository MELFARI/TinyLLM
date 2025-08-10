# --- Imports ---
import os
import requests
import logging
import json
from tqdm import tqdm
from google.colab import drive #Save checpoints, if they dont save when Colab gets disconnected
drive.mount('/content/drive')
# --- API Key Setup ---
# Set your API key securely (you can paste it here, but better use Colab's "Secrets" if possible)
os.environ["OPENROUTER_API_KEY"] = ""

#Loading in the Professor(GPT-OSS-12Ob) and TA(Qwen3-4B)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set your OpenRouter API key as OPENROUTER_API_KEY environment variable")

# --- Base URL ---
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# --- Model IDs ---
PROFESSOR_MODEL = "openai/gpt-oss-120b"     # Professor
TA_MODEL = "qwen/qwen3-4b:free"               # TA
def query_openrouter(model_id, messages):
    """Send a request to OpenRouter model API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 512
    }
    print("Request payload:", payload) # Add print statement here
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")  # See exactly what server returned

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"],
    
    def create_kd_dataset(prompts, output_path="kd_dataset.jsonl"):
    """
    Create a dataset of teacher outputs for KD training.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in tqdm(prompts, desc="Generating teacher outputs"):
            teacher_response = query_openrouter(PROFESSOR_MODEL, [{"role": "user", "content": prompt}])
            if teacher_response is None:
                print(f"Skipping prompt due to no response: {prompt}")
                continue
            json.dump({"prompt": prompt, "teacher_output": teacher_response}, f)
            f.write("\n")
    logging.info(f"KD dataset saved to {output_path}")
    def evaluate_ta(kd_dataset_path):
    """
    Evaluate TA by feeding the same prompts and comparing to teacher.
    """
    with open(kd_dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    for line in tqdm(lines, desc="Evaluating TA"):
        data = json.loads(line)
        prompt = data["prompt"]
        ta_response = query_openrouter(TA_MODEL, [{"role": "user", "content": prompt}])
        print(f"\nPrompt: {prompt}")
        print(f"Teacher: {data['teacher_output']}")
        print(f"TA: {ta_response}")
