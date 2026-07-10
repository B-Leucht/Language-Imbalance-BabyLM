import os
from pathlib import Path
from huggingface_hub import HfApi

import modal

app = modal.App("babylm-upload-existing")

volume = modal.Volume.from_name("babylm-teff-results", create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
)

@app.function(
    image=image,
    volumes={"/results": volume},
    secrets=[hf_secret],
    timeout=2 * 60 * 60,
)
def upload_existing_runs():
    #add token
    api = HfApi(token=os.environ["HF_TOKEN"])
    #add user
    hf_username = os.environ.get("HF_USERNAME")
    if not hf_username:
        raise ValueError("HF_USERNAME is not set. Add it to the Modal secret.")
    
    runs = [
        ("large-ind-1m-8ep-2e-4-32b",  "ind-1m-8ep-2e-4-32b"),
        ("large-ind-3m-8ep-2e-4-32b",  "ind-3m-8ep-2e-4-32b"),
        ("large-ind-10m-8ep-2e-4-32b", "ind-10m-8ep-2e-4-32b"),
        ("large-ind-30m-8ep-2e-4-32b", "ind-30m-8ep-2e-4-32b"),
        ("large-ind-50m-8ep-2e-4-32b", "ind-50m-8ep-2e-4-32b"),
    ]

    for local_name, repo_name in runs:
        folder_path = Path("/results") / local_name

        if not folder_path.exists():
            print("Missing:", folder_path)
            continue

        repo_id = f"{hf_username}/{repo_name}"

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )

        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
        )

        print("Uploaded:", repo_id)

@app.local_entrypoint()
def main():
    upload_existing_runs.remote()