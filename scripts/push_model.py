import os
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi()
repo_id = "ximbor/sentiment-monitor"

try:
    api.create_repo(repo_id=repo_id, token=HF_TOKEN, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Could not create the repository: {e}")

api.upload_folder(
    folder_path="../tmp_model",
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN
)
print(f"Model loaded to '{repo_id}'")