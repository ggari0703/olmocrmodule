import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.getenv("HF_TOKEN")  # 위에서 export한 값 사용

REPOS = [
    "allenai/olmOCR-2-7B-1025",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

BASE_DIR = "./models"

for repo_id in REPOS:
    local_dir = f"{BASE_DIR}/{repo_id.replace('/', '__')}"
    print(f"\n📥 Downloading {repo_id} → {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,             
        resume_download=True,       
        max_workers=8,              
    )

print("\n All done.")
